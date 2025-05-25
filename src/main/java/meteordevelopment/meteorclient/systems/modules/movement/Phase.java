/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.network.packet.c2s.common.CommonPongC2SPacket;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.network.packet.c2s.play.TeleportConfirmC2SPacket;
import net.minecraft.network.packet.s2c.common.KeepAliveS2CPacket;
import net.minecraft.network.packet.s2c.play.PlayerPositionLookS2CPacket;
import net.minecraft.util.math.Vec3d;

public class Phase extends Module {
    private int teleportId;
    private long transactionId;
    private boolean phasing;
    private Vec3d phasePos;
    private int ticks;

    public Phase() {
        super(Categories.Movement, "phase", "Attempts to clip through thin walls using teleport desyncs.");
    }

    @EventHandler
    private void onReceive(PacketEvent.Receive event) {
        if (event.packet instanceof PlayerPositionLookS2CPacket packet) {
            teleportId = packet.teleportId();
            phasePos = packet.change().position();
            phasing = true;
            ticks = 0;
        } else if (event.packet instanceof KeepAliveS2CPacket packet) {
            transactionId = packet.getId();
        }
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (!phasing || mc.getNetworkHandler() == null) return;

        mc.getNetworkHandler().sendPacket(new CommonPongC2SPacket((int) transactionId));
        mc.getNetworkHandler().sendPacket(new TeleportConfirmC2SPacket(teleportId));

        Vec3d dir = mc.player.getRotationVecClient();
        phasePos = phasePos.add(dir.x * 0.05, 0, dir.z * 0.05);
        mc.getNetworkHandler().sendPacket(new PlayerMoveC2SPacket.PositionAndOnGround(
            phasePos.x, phasePos.y, phasePos.z, true, mc.player.horizontalCollision
        ));

        ticks++;
        if (ticks > 5) phasing = false;
    }
}
