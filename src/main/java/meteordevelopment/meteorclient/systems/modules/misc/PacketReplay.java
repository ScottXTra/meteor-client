/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.gui.GuiTheme;
import meteordevelopment.meteorclient.gui.widgets.WLabel;
import meteordevelopment.meteorclient.gui.widgets.WWidget;
import meteordevelopment.meteorclient.gui.widgets.containers.WHorizontalList;
import meteordevelopment.meteorclient.gui.widgets.containers.WTable;
import meteordevelopment.meteorclient.gui.widgets.containers.WVerticalList;
import meteordevelopment.meteorclient.gui.widgets.containers.WView;
import meteordevelopment.meteorclient.gui.widgets.pressable.WButton;
import meteordevelopment.meteorclient.gui.widgets.pressable.WMinus;
import meteordevelopment.meteorclient.gui.widgets.pressable.WPlus;
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
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class PacketReplay extends Module {
    private static final double POSITION_INCREMENT = 0.1;

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
    public WWidget getWidget(GuiTheme theme) {
        WView view = theme.view();

        WLabel countLabel = view.add(theme.label(packetCountText())).expandX().widget();

        WTable table = view.add(theme.table()).expandX().widget();
        table.horizontalSpacing = 6;
        table.verticalSpacing = 4;
        rebuildPacketTable(theme, table);

        WButton refresh = view.add(theme.button("Refresh list")).expandX().widget();
        refresh.action = () -> {
            countLabel.set(packetCountText());
            rebuildPacketTable(theme, table);
        };

        return view;
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
            if (entry.getTimestamp() > elapsed) break;

            if (entry.isMovement()) sendMovement(entry.getMovement());
            else mc.getNetworkHandler().sendPacket(entry.getPacket());

            replayIndex++;
        }

        if (replayIndex >= recordedPackets.size()) {
            replaying = false;
            toggle();
        }
    }

    private void sendMovement(MovementData movement) {
        if (mc.getNetworkHandler() == null) return;

        Vec3d absolutePos = recordedOrigin.add(movement.getRelativePos()).add(replayOffset);
        Packet<?> packet;

        if (movement.hasPosition() && movement.hasRotation()) {
            packet = new PlayerMoveC2SPacket.Full(
                absolutePos.x,
                absolutePos.y,
                absolutePos.z,
                movement.getYaw(),
                movement.getPitch(),
                movement.isOnGround(),
                movement.hasHorizontalCollision()
            );
        } else if (movement.hasPosition()) {
            packet = new PlayerMoveC2SPacket.PositionAndOnGround(
                absolutePos.x,
                absolutePos.y,
                absolutePos.z,
                movement.isOnGround(),
                movement.hasHorizontalCollision()
            );
        } else if (movement.hasRotation()) {
            packet = new PlayerMoveC2SPacket.LookAndOnGround(
                movement.getYaw(),
                movement.getPitch(),
                movement.isOnGround(),
                movement.hasHorizontalCollision()
            );
        } else {
            packet = new PlayerMoveC2SPacket.OnGroundOnly(
                movement.isOnGround(),
                movement.hasHorizontalCollision()
            );
        }

        if (movement.hasPosition()) serverPos = absolutePos;

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

    private void rebuildPacketTable(GuiTheme theme, WTable table) {
        table.clear();

        List<RecordedPacket> movementPackets = new ArrayList<>();
        for (RecordedPacket recordedPacket : recordedPackets) {
            if (recordedPacket.isMovement()) movementPackets.add(recordedPacket);
        }

        if (movementPackets.isEmpty()) {
            table.add(theme.label("No recorded movement packets."));
            table.row();
            return;
        }

        table.add(theme.label("#"));
        table.add(theme.label("Timestamp"));
        table.add(theme.label("Type")).expandCellX();
        table.add(theme.label("Position")).expandCellX();
        table.row();

        for (int i = 0; i < movementPackets.size(); i++) {
            RecordedPacket entry = movementPackets.get(i);

            table.add(theme.label(Integer.toString(i + 1)));
            table.add(theme.label(String.format(Locale.US, "%d ms", entry.getTimestamp())));

            if (entry.isMovement()) {
                MovementData movement = entry.getMovement();
                table.add(theme.label(movement.hasPosition()
                    ? "Movement"
                    : "Movement (no position)")).expandCellX();

                if (movement.hasPosition()) {
                    table.add(createMovementEditor(theme, movement)).expandX();
                } else {
                    table.add(theme.label("Not editable")).expandCellX();
                }
            }

            table.row();
        }
    }

    private WWidget createMovementEditor(GuiTheme theme, MovementData movement) {
        WVerticalList list = theme.verticalList();
        list.spacing = 2;

        list.add(createAxisRow(theme, movement, Direction.Axis.X)).expandX();
        list.add(createAxisRow(theme, movement, Direction.Axis.Y)).expandX();
        list.add(createAxisRow(theme, movement, Direction.Axis.Z)).expandX();

        return list;
    }

    private WWidget createAxisRow(GuiTheme theme, MovementData movement, Direction.Axis axis) {
        WHorizontalList row = theme.horizontalList();
        row.spacing = 4;

        row.add(theme.label(axis.name() + ":")).padRight(4).widget();

        WMinus minus = row.add(theme.minus()).widget();
        WLabel value = row.add(theme.label(formatCoordinate(getAxisAbsoluteValue(movement, axis)))).padHorizontal(4).widget();
        WPlus plus = row.add(theme.plus()).widget();

        minus.action = () -> adjustMovementAxis(movement, axis, -POSITION_INCREMENT, value);
        plus.action = () -> adjustMovementAxis(movement, axis, POSITION_INCREMENT, value);

        return row;
    }

    private void adjustMovementAxis(MovementData movement, Direction.Axis axis, double delta, WLabel valueLabel) {
        movement.adjustPosition(
            axis == Direction.Axis.X ? delta : 0,
            axis == Direction.Axis.Y ? delta : 0,
            axis == Direction.Axis.Z ? delta : 0
        );

        valueLabel.set(formatCoordinate(getAxisAbsoluteValue(movement, axis)));
    }

    private double getAxisAbsoluteValue(MovementData movement, Direction.Axis axis) {
        Vec3d relative = movement.getRelativePos();

        return switch (axis) {
            case X -> recordedOrigin.x + relative.x;
            case Y -> recordedOrigin.y + relative.y;
            case Z -> recordedOrigin.z + relative.z;
        };
    }

    private String formatCoordinate(double value) {
        return String.format(Locale.US, "%.3f", value);
    }

    private String packetCountText() {
        return String.format(Locale.US, "Recorded packets: %d", recordedPackets.size());
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

        public long getTimestamp() {
            return timestamp;
        }

        public Packet<?> getPacket() {
            return packet;
        }

        public MovementData getMovement() {
            return movement;
        }

        public boolean isMovement() {
            return movement != null;
        }
    }

    private static class MovementData {
        private Vec3d relativePos;
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

        public Vec3d getRelativePos() {
            return relativePos;
        }

        public boolean hasPosition() {
            return hasPosition;
        }

        public boolean hasRotation() {
            return hasRotation;
        }

        public float getYaw() {
            return yaw;
        }

        public float getPitch() {
            return pitch;
        }

        public boolean isOnGround() {
            return onGround;
        }

        public boolean hasHorizontalCollision() {
            return horizontalCollision;
        }

        public void adjustPosition(double x, double y, double z) {
            if (!hasPosition) return;
            relativePos = relativePos.add(x, y, z);
        }
    }

    private enum Mode {
        Record,
        Replay
    }
}
