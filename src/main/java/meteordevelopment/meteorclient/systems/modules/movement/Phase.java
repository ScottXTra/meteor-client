/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.network.packet.c2s.play.TeleportConfirmC2SPacket;
import net.minecraft.network.packet.s2c.play.PlayerPositionLookS2CPacket;

/**
 * Attempts to clip through blocks by abusing teleport confirmation.
 */
public class Phase extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> offset = sgGeneral.add(new DoubleSetting.Builder()
        .name("offset")
        .description("Horizontal offset applied when confirming the teleport.")
        .defaultValue(0.05)
        .min(-1)
        .sliderRange(-1, 1)
        .build()
    );

    private final Setting<Boolean> autoDisable = sgGeneral.add(new BoolSetting.Builder()
        .name("auto-disable")
        .description("Disable after phasing once.")
        .defaultValue(true)
        .build()
    );

    private boolean awaitingTeleport = false;
    private int teleportId = -1;

    public Phase() {
        super(Categories.Movement, "phase", "Attempts to move through thin walls using teleport packets.");
    }

    @Override
    public void onActivate() {
        if (mc.player == null) return;

        awaitingTeleport = true;
        teleportId = -1;

        // Send an invalid position to trigger a correction from the server
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.Full(
            mc.player.getX(), mc.player.getY() + 5, mc.player.getZ(),
            mc.player.getYaw(), mc.player.getPitch(), mc.player.isOnGround(),
            mc.player.horizontalCollision
        ));
    }

    @Override
    public void onDeactivate() {
        awaitingTeleport = false;
        teleportId = -1;
    }

    @EventHandler
    private void onReceive(PacketEvent.Receive event) {
        if (!awaitingTeleport) return;
        if (!(event.packet instanceof PlayerPositionLookS2CPacket packet)) return;

        teleportId = packet.teleportId();

        // Cancel original packet to prevent server from moving us automatically
        event.cancel();

        double off = offset.get();

        // Confirm teleport
        mc.player.networkHandler.sendPacket(new TeleportConfirmC2SPacket(teleportId));

        // Move slightly off the teleported position
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.Full(
            packet.change().position().x + off,
            packet.change().position().y,
            packet.change().position().z + off,
            packet.change().yaw(),
            packet.change().pitch(),
            false,
            mc.player.horizontalCollision
        ));

        awaitingTeleport = false;
        if (autoDisable.get()) toggle();
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (awaitingTeleport || teleportId == -1) return;

        // Continue sending movement packets to push the player through
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.PositionAndOnGround(
            mc.player.getX(), mc.player.getY(), mc.player.getZ(), mc.player.isOnGround(),
            mc.player.horizontalCollision
        ));
    }
}
