package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.PlayerPosition;
import net.minecraft.network.packet.Packet;
import net.minecraft.network.listener.ClientPlayPacketListener;
import net.minecraft.network.packet.s2c.play.EntityPositionS2CPacket;
import net.minecraft.network.packet.s2c.play.EntityS2CPacket;
import net.minecraft.util.math.Box;
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

    private final Setting<SettingColor> realColor = sgGeneral.add(new ColorSetting.Builder()
        .name("real-color")
        .description("Color of the real position box.")
        .defaultValue(new SettingColor(255, 0, 0, 75))
        .build()
    );

    private final Queue<DelayedPacket> packets = new ConcurrentLinkedQueue<>();
    private final Map<Integer, Vec3d> realPositions = new ConcurrentHashMap<>();

    public BackTrack() {
        super(Categories.Combat, "backtrack", "Delays incoming movement packets making players appear at previous positions.");
    }

    @Override
    public void onDeactivate() {
        packets.clear();
        realPositions.clear();
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

            packets.add(new DelayedPacket(packet, entity.getId(), System.currentTimeMillis() + randomDelay()));
            event.cancel();
        } else if (packet instanceof EntityPositionS2CPacket p) {
            Entity entity = mc.world.getEntityById(p.entityId());
            if (!shouldTrack(entity)) return;

            PlayerPosition applied = PlayerPosition.apply(PlayerPosition.fromEntity(entity), p.change(), p.relatives());
            realPositions.put(entity.getId(), applied.position());

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

        // cleanup positions
        realPositions.entrySet().removeIf(entry -> {
            Entity e = mc.world.getEntityById(entry.getKey());
            return e == null || e.distanceTo(mc.player) > range.get();
        });
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        for (Map.Entry<Integer, Vec3d> entry : realPositions.entrySet()) {
            Entity entity = mc.world.getEntityById(entry.getKey());
            if (entity == null) continue;

            Vec3d pos = entry.getValue();
            double w = entity.getWidth();
            double h = entity.getHeight();
            Box box = new Box(pos.x - w / 2, pos.y, pos.z - w / 2, pos.x + w / 2, pos.y + h, pos.z + w / 2);
            event.renderer.box(box, realColor.get(), realColor.get(), ShapeMode.Lines, 0);
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