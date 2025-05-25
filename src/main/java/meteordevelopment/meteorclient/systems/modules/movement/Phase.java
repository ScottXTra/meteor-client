/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.IntSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.player.PlayerPosition;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.network.packet.c2s.play.TeleportConfirmC2SPacket;
import net.minecraft.network.packet.s2c.play.PlayerPositionLookS2CPacket;
import net.minecraft.network.packet.s2c.play.PositionFlag;

import java.util.Set;

public class Phase extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> offset = sgGeneral.add(new DoubleSetting.Builder()
        .name("offset")
        .description("Horizontal offset applied when confirming the teleport.")
        .defaultValue(0.3)
        .min(0)
        .sliderMax(1)
        .build()
    );

    private final Setting<Integer> pushTicks = sgGeneral.add(new IntSetting.Builder()
        .name("push-ticks")
        .description("Number of ticks to keep sending movement packets after teleport.")
        .defaultValue(3)
        .min(0)
        .sliderMax(20)
        .build()
    );

    private boolean phasing;
    private int ticksLeft;
    private float yaw;
    private float pitch;

    public Phase() {
        super(Categories.Movement, "phase", "Attempts to clip you through thin blocks.");
    }

    @EventHandler
    private void onReceive(PacketEvent.Receive event) {
        if (!(event.packet instanceof PlayerPositionLookS2CPacket packet)) return;

        event.cancel();

        PlayerPosition pos = packet.change();
        Set<PositionFlag> flags = packet.relatives();

        double x = flags.contains(PositionFlag.X) ? mc.player.getX() + pos.position().getX() : pos.position().getX();
        double y = flags.contains(PositionFlag.Y) ? mc.player.getY() + pos.position().getY() : pos.position().getY();
        double z = flags.contains(PositionFlag.Z) ? mc.player.getZ() + pos.position().getZ() : pos.position().getZ();

        yaw = flags.contains(PositionFlag.Y_ROT) ? mc.player.getYaw() + pos.yaw() : pos.yaw();
        pitch = flags.contains(PositionFlag.X_ROT) ? mc.player.getPitch() + pos.pitch() : pos.pitch();

        double rad = Math.toRadians(yaw);
        double dx = Math.cos(rad) * offset.get();
        double dz = Math.sin(rad) * offset.get();

        x += dx;
        z += dz;

        mc.player.setPosition(x, y, z);

        mc.player.networkHandler.sendPacket(new TeleportConfirmC2SPacket(packet.teleportId()));
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.Full(x, y, z, yaw, pitch, false, false));

        phasing = true;
        ticksLeft = pushTicks.get();
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (!phasing || ticksLeft <= 0) return;

        double rad = Math.toRadians(yaw);
        double dx = Math.cos(rad) * offset.get();
        double dz = Math.sin(rad) * offset.get();

        double x = mc.player.getX() + dx;
        double y = mc.player.getY();
        double z = mc.player.getZ() + dz;

        mc.player.setPosition(x, y, z);
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.PositionAndOnGround(x, y, z, mc.player.isOnGround(), false));

        ticksLeft--;
        if (ticksLeft <= 0) phasing = false;
    }
}
