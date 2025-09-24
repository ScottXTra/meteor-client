/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.PacketListSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.network.PacketUtils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.network.packet.Packet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.Set;

/**
 * Logs packets to a file and chat.
 */
public class PacketLogger extends Module {
    private BufferedWriter writer;
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
    private static final Map<String, String> FIELD_NAME_OVERRIDES = Map.ofEntries(
        Map.entry("field_12889", "x"),
        Map.entry("field_12886", "y"),
        Map.entry("field_12884", "z"),
        Map.entry("field_12887", "yaw"),
        Map.entry("field_12885", "pitch"),
        Map.entry("field_29179", "onGround"),
        Map.entry("field_52335", "horizontalCollision"),
        Map.entry("field_12890", "changePosition"),
        Map.entry("field_12888", "changeLook")
    );

    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Set<Class<? extends Packet<?>>>> ignorePackets = sgGeneral.add(new PacketListSetting.Builder()
        .name("ignore-packets")
        .description("Client-to-server packets to not log.")
        .filter(aClass -> PacketUtils.getC2SPackets().contains(aClass))
        .build()
    );

    private final Setting<Boolean> logIncoming = sgGeneral.add(new BoolSetting.Builder()
        .name("log-incoming")
        .description("Logs packets received from the server.")
        .defaultValue(false)
        .build()
    );

    private final Setting<Set<Class<? extends Packet<?>>>> ignoreIncomingPackets = sgGeneral.add(new PacketListSetting.Builder()
        .name("ignore-incoming-packets")
        .description("Server-to-client packets to not log.")
        .filter(aClass -> PacketUtils.getS2CPackets().contains(aClass))
        .visible(logIncoming::get)
        .build()
    );

    public PacketLogger() {
        super(Categories.Misc, "packet-logger", "Logs packets to a file and chat.");
    }

    @Override
    public void onActivate() {
        try {
            File folder = new File(MeteorClient.FOLDER, "packet-logs");
            folder.mkdirs();

            String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            File file = new File(folder, "packets-" + time + ".txt");

            writer = new BufferedWriter(new FileWriter(file));
            info("Logging packets to %s", file.getName());
        } catch (IOException e) {
            error("Failed to create log file.");
            e.printStackTrace();
            toggle();
        }
    }

    @Override
    public void onDeactivate() {
        if (writer != null) {
            try {
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @EventHandler
    private void onSendPacket(PacketEvent.Send event) {
        @SuppressWarnings("unchecked")
        Class<? extends Packet<?>> packetClass = (Class<? extends Packet<?>>) event.packet.getClass();
        if (ignorePackets.get().contains(packetClass)) return;

        String name = PacketUtils.getName(packetClass);
        if (name == null) name = packetClass.getName();

        StringBuilder logBuilder = new StringBuilder();
        logBuilder.append("[").append(LocalDateTime.now().format(TIME_FORMATTER)).append("] ");
        logBuilder.append(name);

        Packet<?> packet = event.packet;
        Class<?> clazz = packet.getClass();
        while (clazz != Object.class) {
            for (Field field : clazz.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers())) continue;
                field.setAccessible(true);
                try {
                    Object value = field.get(packet);
                    String fieldName = FIELD_NAME_OVERRIDES.getOrDefault(field.getName(), field.getName());
                    logBuilder.append(" ").append(fieldName).append("=").append(value);
                } catch (IllegalAccessException ignored) {
                }
            }
            clazz = clazz.getSuperclass();
        }

        String log = logBuilder.toString();
        info(log);

        if (writer != null) {
            try {
                writer.write(log);
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @EventHandler
    private void onReceivePacket(PacketEvent.Receive event) {
        if (!logIncoming.get()) return;

        @SuppressWarnings("unchecked")
        Class<? extends Packet<?>> packetClass = (Class<? extends Packet<?>>) event.packet.getClass();
        if (ignoreIncomingPackets.get().contains(packetClass)) return;

        String name = PacketUtils.getName(packetClass);
        if (name == null) name = packetClass.getName();

        StringBuilder logBuilder = new StringBuilder();
        logBuilder.append("[").append(LocalDateTime.now().format(TIME_FORMATTER)).append("] ");
        logBuilder.append(name);

        Packet<?> packet = event.packet;
        Class<?> clazz = packet.getClass();
        while (clazz != Object.class) {
            for (Field field : clazz.getDeclaredFields()) {
                if (Modifier.isStatic(field.getModifiers())) continue;
                field.setAccessible(true);
                try {
                    Object value = field.get(packet);
                    String fieldName = FIELD_NAME_OVERRIDES.getOrDefault(field.getName(), field.getName());
                    logBuilder.append(" ").append(fieldName).append("=").append(value);
                } catch (IllegalAccessException ignored) {
                }
            }
            clazz = clazz.getSuperclass();
        }

        String log = logBuilder.toString();
        info(log);

        if (writer != null) {
            try {
                writer.write(log);
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
