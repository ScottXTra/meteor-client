/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.EnumSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.network.packet.Packet;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.util.ArrayList;
import java.util.List;

public class PacketReplay extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Mode> mode = sgGeneral.add(new EnumSetting.Builder<Mode>()
        .name("mode")
        .description("Whether to record outgoing packets or replay the last recording.")
        .defaultValue(Mode.Record)
        .onChanged(this::onModeChanged)
        .build()
    );

    private final Setting<Boolean> renderServerBox = sgGeneral.add(new BoolSetting.Builder()
        .name("render-server-box")
        .description("Render the server position while replaying.")
        .defaultValue(true)
        .build()
    );

    private final List<RecordedPacket> recordedPackets = new ArrayList<>();

    private Vec3d recordedOrigin = Vec3d.ZERO;
    private Vec3d recordServerPos = Vec3d.ZERO;
    private float recordYaw;
    private float recordPitch;
    private long recordStartTime;

    private boolean recording;

    private boolean replaying;
    private long replayStartTime;
    private int replayIndex;
    private Vec3d replayOffset = Vec3d.ZERO;
    private Vec3d serverPos = Vec3d.ZERO;

    private Mode activeMode = Mode.Record;

    public PacketReplay() {
        super(Categories.Misc, "packet-replay", "Record outgoing packets and replay them later.");
    }

    @Override
    public void onActivate() {
        activeMode = mode.get();
        if (activeMode == Mode.Record) startRecording();
        else startReplay();
    }

    @Override
    public void onDeactivate() {
        if (recording) finishRecording();
        recording = false;
        replaying = false;
    }

    private void onModeChanged(Mode mode) {
        if (!isActive()) {
            activeMode = mode;
            return;
        }

        if (recording) finishRecording();
        recording = false;
        replaying = false;
        activeMode = mode;

        if (mode == Mode.Record) startRecording();
        else startReplay();
    }

    private void startRecording() {
        if (mc.player == null) {
            error("Cannot record without a player.");
            toggle();
            return;
        }

        recordedPackets.clear();
        recordStartTime = System.currentTimeMillis();
        recordedOrigin = mc.player.getPos();
        recordServerPos = recordedOrigin;
        recordYaw = mc.player.getYaw();
        recordPitch = mc.player.getPitch();
        serverPos = recordedOrigin;
        recording = true;
    }

    private void finishRecording() {
        recording = false;
    }

    private void startReplay() {
        if (recordedPackets.isEmpty()) {
            error("No recording available.");
            toggle();
            return;
        }

        if (mc.player == null || mc.getNetworkHandler() == null) {
            error("Cannot replay without a player.");
            toggle();
            return;
        }

        replayStartTime = System.currentTimeMillis();
        replayIndex = 0;
        Vec3d currentPos = mc.player.getPos();
        replayOffset = new Vec3d(
            currentPos.x - recordedOrigin.x,
            currentPos.y - recordedOrigin.y,
            currentPos.z - recordedOrigin.z
        );
        serverPos = recordedOrigin.add(replayOffset);
        replaying = true;
    }

    @EventHandler
    private void onSendPacket(PacketEvent.Send event) {
        if (!recording) return;

        long timestamp = System.currentTimeMillis() - recordStartTime;
        Packet<?> packet = event.packet;

        if (packet instanceof PlayerMoveC2SPacket move) {
            Vec3d newPos = new Vec3d(
                move.getX(recordServerPos.x),
                move.getY(recordServerPos.y),
                move.getZ(recordServerPos.z)
            );

            Vec3d relativePos = new Vec3d(
                newPos.x - recordedOrigin.x,
                newPos.y - recordedOrigin.y,
                newPos.z - recordedOrigin.z
            );

            float yaw = move.getYaw(recordYaw);
            float pitch = move.getPitch(recordPitch);

            if (move.changesLook()) {
                recordYaw = yaw;
                recordPitch = pitch;
            }

            MovementData movement = new MovementData(
                relativePos,
                move.changesPosition(),
                move.changesLook(),
                yaw,
                pitch,
                move.isOnGround(),
                mc.player != null && mc.player.horizontalCollision
            );

            recordedPackets.add(RecordedPacket.forMovement(timestamp, movement));
            recordServerPos = newPos;
        } else {
            recordedPackets.add(RecordedPacket.forPacket(timestamp, packet));
        }
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (!replaying || mc.getNetworkHandler() == null) return;

        if (mc.player == null) {
            error("Player is unavailable. Stopping replay.");
            toggle();
            return;
        }

        long elapsed = System.currentTimeMillis() - replayStartTime;

        while (replayIndex < recordedPackets.size()) {
            RecordedPacket entry = recordedPackets.get(replayIndex);
            if (entry.timestamp > elapsed) break;

            if (entry.movement != null) sendMovement(entry.movement);
            else mc.getNetworkHandler().sendPacket(entry.packet);

            replayIndex++;
        }

        if (replayIndex >= recordedPackets.size()) {
            replaying = false;
            toggle();
        }
    }

    private void sendMovement(MovementData movement) {
        if (mc.getNetworkHandler() == null) return;

        Vec3d absolutePos = recordedOrigin.add(movement.relativePos).add(replayOffset);
        Packet<?> packet;

        if (movement.hasPosition && movement.hasRotation) {
            packet = new PlayerMoveC2SPacket.Full(
                absolutePos.x,
                absolutePos.y,
                absolutePos.z,
                movement.yaw,
                movement.pitch,
                movement.onGround,
                movement.horizontalCollision
            );
        } else if (movement.hasPosition) {
            packet = new PlayerMoveC2SPacket.PositionAndOnGround(
                absolutePos.x,
                absolutePos.y,
                absolutePos.z,
                movement.onGround,
                movement.horizontalCollision
            );
        } else if (movement.hasRotation) {
            packet = new PlayerMoveC2SPacket.LookAndOnGround(
                movement.yaw,
                movement.pitch,
                movement.onGround,
                movement.horizontalCollision
            );
        } else {
            packet = new PlayerMoveC2SPacket.OnGroundOnly(
                movement.onGround,
                movement.horizontalCollision
            );
        }

        if (movement.hasPosition) serverPos = absolutePos;

        mc.getNetworkHandler().sendPacket(packet);
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (!renderServerBox.get() || !replaying || mc.player == null) return;

        Box box = mc.player.getBoundingBox().offset(
            serverPos.x - mc.player.getX(),
            serverPos.y - mc.player.getY(),
            serverPos.z - mc.player.getZ()
        );

        event.renderer.box(box, Color.RED, Color.RED, ShapeMode.Lines, 0);
    }

    private static class RecordedPacket {
        private final long timestamp;
        private final Packet<?> packet;
        private final MovementData movement;

        private RecordedPacket(long timestamp, Packet<?> packet, MovementData movement) {
            this.timestamp = timestamp;
            this.packet = packet;
            this.movement = movement;
        }

        public static RecordedPacket forPacket(long timestamp, Packet<?> packet) {
            return new RecordedPacket(timestamp, packet, null);
        }

        public static RecordedPacket forMovement(long timestamp, MovementData movement) {
            return new RecordedPacket(timestamp, null, movement);
        }
    }

    private static class MovementData {
        private final Vec3d relativePos;
        private final boolean hasPosition;
        private final boolean hasRotation;
        private final float yaw;
        private final float pitch;
        private final boolean onGround;
        private final boolean horizontalCollision;

        private MovementData(Vec3d relativePos, boolean hasPosition, boolean hasRotation, float yaw, float pitch, boolean onGround, boolean horizontalCollision) {
            this.relativePos = relativePos;
            this.hasPosition = hasPosition;
            this.hasRotation = hasRotation;
            this.yaw = yaw;
            this.pitch = pitch;
            this.onGround = onGround;
            this.horizontalCollision = horizontalCollision;
        }
    }

    private enum Mode {
        Record,
        Replay
    }
}
