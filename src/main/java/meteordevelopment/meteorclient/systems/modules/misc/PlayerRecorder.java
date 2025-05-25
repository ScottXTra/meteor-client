/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.entity.EntityAddedEvent;
import meteordevelopment.meteorclient.events.entity.EntityRemovedEvent;
import meteordevelopment.meteorclient.events.entity.LivingEntityMoveEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.PlayerListSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.player.PlayerEntity;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class PlayerRecorder extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Set<String> visiblePlayers = new HashSet<>();

    private final Setting<List<String>> players = sgGeneral.add(new PlayerListSetting.Builder()
        .name("players")
        .description("Players to record.")
        .supplier(() -> visiblePlayers)
        .build());

    private final Setting<Boolean> recordAll = sgGeneral.add(new BoolSetting.Builder()
        .name("record-all")
        .description("Record every player in render distance.")
        .defaultValue(false)
        .build());

    private final Map<String, BufferedWriter> writers = new HashMap<>();
    private File folder;

    public PlayerRecorder() {
        super(Categories.Misc, "player-recorder", "Records other players' positions to text files.");
    }

    @Override
    public void onActivate() {
        folder = new File(MeteorClient.FOLDER, "player-positions");
        folder.mkdirs();
        visiblePlayers.clear();
        writers.clear();

        for (PlayerEntity player : mc.world.getPlayers()) {
            if (player == mc.player) continue;
            visiblePlayers.add(player.getName().getString());
            if (recordAll.get() || players.get().contains(player.getName().getString())) start(player.getName().getString());
        }
    }

    @Override
    public void onDeactivate() {
        for (BufferedWriter writer : writers.values()) {
            try { writer.close(); } catch (IOException ignored) {}
        }
        writers.clear();
        visiblePlayers.clear();
    }

    @EventHandler
    private void onEntityAdded(EntityAddedEvent event) {
        if (!(event.entity instanceof PlayerEntity player) || player == mc.player) return;
        String name = player.getName().getString();
        visiblePlayers.add(name);
        if (recordAll.get() || players.get().contains(name)) start(name);
    }

    @EventHandler
    private void onEntityRemoved(EntityRemovedEvent event) {
        if (!(event.entity instanceof PlayerEntity player) || player == mc.player) return;
        visiblePlayers.remove(player.getName().getString());
    }

    @EventHandler
    private void onMove(LivingEntityMoveEvent event) {
        if (!(event.entity instanceof PlayerEntity player) || player == mc.player) return;
        String name = player.getName().getString();
        if (!recordAll.get() && !players.get().contains(name)) return;

        start(name);
        BufferedWriter writer = writers.get(name);
        if (writer == null) return;

        try {
            writer.write(String.format(Locale.US, "%f %f %f%n", player.getX(), player.getY(), player.getZ()));
            writer.flush();
        } catch (IOException ignored) { }
    }

    private void start(String name) {
        if (writers.containsKey(name)) return;
        try {
            File file = new File(folder, name + ".txt");
            writers.put(name, new BufferedWriter(new FileWriter(file, true)));
        } catch (IOException ignored) { }
    }
}
