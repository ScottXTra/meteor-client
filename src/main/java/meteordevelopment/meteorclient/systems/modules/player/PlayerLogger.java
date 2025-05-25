/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.player;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.PlayerListSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.Utils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.Vec3d;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class PlayerLogger extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Set<String>> players = sgGeneral.add(new PlayerListSetting.Builder()
        .name("players")
        .description("Players to record.")
        .build()
    );

    private final Map<String, BufferedWriter> writers = new HashMap<>();
    private File folder;

    public PlayerLogger() {
        super(Categories.Player, "player-logger", "Records selected players' positions to files.");
    }

    @Override
    public void onActivate() {
        folder = new File(new File(meteordevelopment.meteorclient.MeteorClient.FOLDER, "player-positions"), Utils.getFileWorldName());
        folder.mkdirs();
    }

    @Override
    public void onDeactivate() {
        for (BufferedWriter writer : writers.values()) {
            try {
                writer.close();
            } catch (IOException ignored) {
            }
        }
        writers.clear();
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (mc.world == null) return;

        for (PlayerEntity player : mc.world.getPlayers()) {
            if (player == mc.player) continue;
            String name = player.getGameProfile().getName();
            if (!players.get().contains(name)) continue;

            BufferedWriter writer = writers.get(name);
            if (writer == null) {
                try {
                    File file = new File(folder, sanitize(name) + ".txt");
                    writer = new BufferedWriter(new FileWriter(file, true));
                    writers.put(name, writer);
                } catch (IOException e) {
                    e.printStackTrace();
                    continue;
                }
            }

            Vec3d pos = player.getPos();
            try {
                writer.write(String.format("%.2f %.2f %.2f%n", pos.x, pos.y, pos.z));
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static String sanitize(String name) {
        return name.replaceAll("[^a-zA-Z0-9-_]", "_");
    }
}
