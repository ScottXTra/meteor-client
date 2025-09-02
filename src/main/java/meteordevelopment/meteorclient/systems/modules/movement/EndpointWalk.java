/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import meteordevelopment.orbit.EventPriority;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.network.ClientConnection;
import net.minecraft.network.packet.Packet;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class EndpointWalk extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> length = sgGeneral.add(new DoubleSetting.Builder()
        .name("length")
        .description("Distance in blocks from the starting point to each endpoint.")
        .defaultValue(5.0)
        .min(1.0)
        .build()
    );

    private final Setting<Double> radius = sgGeneral.add(new DoubleSetting.Builder()
        .name("spoof-radius")
        .description("Radius around other players to enable ping spoofing.")
        .defaultValue(6.0)
        .min(1.0)
        .build()
    );

    private double x1, x2;
    private boolean toFirst;
    private boolean spoofing;
    private Vec3d serverPos = Vec3d.ZERO;

    private final Queue<QueuedPacket> queue = new ConcurrentLinkedQueue<>();

    public EndpointWalk() {
        super(Categories.Movement, "endpoint-walk", "Faces alternating endpoints along the X axis so you can walk back and forth.");
    }

    @Override
    public void onActivate() {
        double x = mc.player.getX();
        x1 = x + length.get();
        x2 = x - length.get();
        toFirst = true;
        spoofing = false;
        serverPos = mc.player.getPos();
    }

    @Override
    public void onDeactivate() {
        flushQueue();
        spoofing = false;
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        double targetX = toFirst ? x1 : x2;

        Vec3d target = new Vec3d(targetX, mc.player.getY(), mc.player.getZ());
        float yaw = (float) Rotations.getYaw(target);
        mc.player.setYaw(yaw);
        mc.player.headYaw = yaw;
        mc.player.bodyYaw = yaw;

        if (Math.abs(mc.player.getX() - targetX) <= 1) {
            toFirst = !toFirst;
        }

        boolean inRange = mc.world.getPlayers().stream()
            .anyMatch(p -> p != mc.player && p.squaredDistanceTo(mc.player) <= radius.get() * radius.get());

        if (inRange) {
            spoofing = true;
        } else if (spoofing) {
            flushQueue();
            spoofing = false;
        }

        if (!spoofing) serverPos = mc.player.getPos();
    }

    @EventHandler(priority = EventPriority.HIGHEST + 50)
    private void onSendPacket(PacketEvent.Send event) {
        if (!spoofing) return;

        queue.add(new QueuedPacket(event.packet, event.connection));
        event.cancel();
    }

    private void flushQueue() {
        QueuedPacket p;
        while ((p = queue.poll()) != null) {
            p.connection.send(p.packet, null, true);
        }
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (mc.player == null || mc.world == null) return;

        double r = radius.get();
        for (PlayerEntity p : mc.world.getPlayers()) {
            renderCircle(event, p.getX(), p.getY(), p.getZ(), r, Color.RED);
        }

        Box box = mc.player.getBoundingBox().offset(
            serverPos.x - mc.player.getX(),
            serverPos.y - mc.player.getY(),
            serverPos.z - mc.player.getZ()
        );

        event.renderer.box(box, Color.RED, Color.RED, ShapeMode.Lines, 0);
    }

    private void renderCircle(Render3DEvent event, double x, double y, double z, double r, Color color) {
        double prevX = x + r;
        double prevZ = z;
        for (int i = 1; i <= 40; i++) {
            double angle = Math.PI * 2 * i / 40;
            double nextX = x + Math.cos(angle) * r;
            double nextZ = z + Math.sin(angle) * r;
            event.renderer.line(prevX, y, prevZ, nextX, y, nextZ, color);
            prevX = nextX;
            prevZ = nextZ;
        }
    }

    private static class QueuedPacket {
        final Packet<?> packet;
        final ClientConnection connection;

        QueuedPacket(Packet<?> packet, ClientConnection connection) {
            this.packet = packet;
            this.connection = connection;
        }
    }
}

