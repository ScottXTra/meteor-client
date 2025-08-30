package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.entity.fakeplayer.FakePlayerEntity;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerPosition;
import net.minecraft.network.listener.ClientPlayPacketListener;
import net.minecraft.network.packet.Packet;
import net.minecraft.network.packet.s2c.play.EntityPositionS2CPacket;
import net.minecraft.network.packet.s2c.play.EntityS2CPacket;
import net.minecraft.util.math.Vec3d;

import java.util.Iterator;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.ConcurrentHashMap;

public class BackTrack extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Integer> minDelay = sgGeneral.add(new IntSetting.Builder()
        .name("min-delay")
        .description("Minimum delay in ms applied to movement packets.")
        .defaultValue(100)
        .min(0)
        .sliderMax(1000)
        .build()
    );

    private final Setting<Integer> maxDelay = sgGeneral.add(new IntSetting.Builder()
        .name("max-delay")
        .description("Maximum delay in ms applied to movement packets.")
        .defaultValue(200)
        .min(0)
        .sliderMax(1000)
        .build()
    );

    private final Setting<Double> range = sgGeneral.add(new DoubleSetting.Builder()
        .name("range")
        .description("Range within which players are back tracked.")
        .defaultValue(10.0)
        .min(1.0)
        .sliderMax(50.0)
        .build()
    );

    private final Setting<SettingColor> lineColor = sgGeneral.add(new ColorSetting.Builder()
        .name("line-color")
        .description("Color of the line connecting real and delayed positions.")
        .defaultValue(new SettingColor(255, 0, 0, 75))
        .build()
    );

    private final Queue<DelayedPacket> packets = new ConcurrentLinkedQueue<>();
    private final Map<Integer, Vec3d> realPositions = new ConcurrentHashMap<>();
    private final Map<Integer, FakePlayerEntity> fakePlayers = new ConcurrentHashMap<>();

    public BackTrack() {
        super(Categories.Combat, "backtrack", "Delays incoming movement packets making players appear at previous positions.");
    }

    @Override
    public void onDeactivate() {
        packets.clear();
        realPositions.clear();
        fakePlayers.values().forEach(FakePlayerEntity::despawn);
        fakePlayers.clear();
    }

    @EventHandler
    private void onReceive(PacketEvent.Receive event) {
        if (mc.world == null || mc.player == null) return;

        Packet<?> packet = event.packet;

        if (packet instanceof EntityS2CPacket s2c) {
            Entity entity = s2c.getEntity(mc.world);
            if (!shouldTrack(entity)) return;

            Vec3d pos = realPositions.getOrDefault(entity.getId(), entity.getPos());
            pos = pos.add(s2c.getDeltaX() / 4096.0, s2c.getDeltaY() / 4096.0, s2c.getDeltaZ() / 4096.0);
            realPositions.put(entity.getId(), pos);

            updateFakePlayer(entity, pos);

            packets.add(new DelayedPacket(packet, entity.getId(), System.currentTimeMillis() + randomDelay()));
            event.cancel();
        } else if (packet instanceof EntityPositionS2CPacket p) {
            Entity entity = mc.world.getEntityById(p.entityId());
            if (!shouldTrack(entity)) return;

            PlayerPosition applied = PlayerPosition.apply(PlayerPosition.fromEntity(entity), p.change(), p.relatives());
            Vec3d pos = applied.position();
            realPositions.put(entity.getId(), pos);

            updateFakePlayer(entity, pos);

            packets.add(new DelayedPacket(packet, entity.getId(), System.currentTimeMillis() + randomDelay()));
            event.cancel();
        }
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        long now = System.currentTimeMillis();
        Iterator<DelayedPacket> it = packets.iterator();
        while (it.hasNext()) {
            DelayedPacket dp = it.next();
            if (now >= dp.time) {
                ((Packet<ClientPlayPacketListener>) dp.packet).apply(mc.player.networkHandler);
                it.remove();
            }
        }

        // cleanup positions and fake players
        realPositions.entrySet().removeIf(entry -> {
            Entity e = mc.world.getEntityById(entry.getKey());
            if (e == null || e.distanceTo(mc.player) > range.get()) {
                FakePlayerEntity fp = fakePlayers.remove(entry.getKey());
                if (fp != null) fp.despawn();
                return true;
            }
            return false;
        });
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        for (Map.Entry<Integer, FakePlayerEntity> entry : fakePlayers.entrySet()) {
            Entity entity = mc.world.getEntityById(entry.getKey());
            if (entity == null) continue;

            FakePlayerEntity fake = entry.getValue();
            event.renderer.line(fake.getX(), fake.getY(), fake.getZ(), entity.getX(), entity.getY(), entity.getZ(), lineColor.get());
        }
    }

    private void updateFakePlayer(Entity entity, Vec3d pos) {
        FakePlayerEntity fake = fakePlayers.get(entity.getId());
        if (fake == null && entity instanceof PlayerEntity player) {
            float health = player.getHealth() + player.getAbsorptionAmount();
            fake = new FakePlayerEntity(player, player.getGameProfile().getName(), health, true);
            fake.doNotPush = true;
            fake.noHit = true;
            fake.spawn();
            fakePlayers.put(entity.getId(), fake);
        }

        if (fake != null) {
            fake.setPos(pos.x, pos.y, pos.z);
            fake.setYaw(entity.getYaw());
            fake.setPitch(entity.getPitch());
            if (entity instanceof PlayerEntity p) {
                fake.headYaw = p.headYaw;
                fake.bodyYaw = p.bodyYaw;
            }
        }
    }

    private boolean shouldTrack(Entity entity) {
        return entity instanceof PlayerEntity player && player != mc.player && mc.player.distanceTo(player) <= range.get();
    }

    private long randomDelay() {
        int min = minDelay.get();
        int max = maxDelay.get();
        if (max <= min) return min;
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }

    private static class DelayedPacket {
        final Packet<?> packet;
        final int entityId;
        final long time;

        DelayedPacket(Packet<?> packet, int entityId, long time) {
            this.packet = packet;
            this.entityId = entityId;
            this.time = time;
        }
    }
}