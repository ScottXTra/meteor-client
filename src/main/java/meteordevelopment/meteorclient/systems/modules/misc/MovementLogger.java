/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Logs player position, velocity and rotation into a CSV file every tick.
 */
public class MovementLogger extends Module {
    private BufferedWriter writer;

    public MovementLogger() {
        super(Categories.Misc, "movement-logger", "Logs position, velocity and head angles to a CSV file.");
    }

    @Override
    public void onActivate() {
        try {
            File folder = new File(MeteorClient.FOLDER, "movement-logs");
            folder.mkdirs();

            String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            File file = new File(folder, "movement-" + time + ".csv");

            writer = new BufferedWriter(new FileWriter(file));
            writer.write("timestamp,x,y,z,vx,vy,vz,pitch,yaw");
            writer.newLine();
            info("Logging movement to %s", file.getName());
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
    private void onTick(TickEvent.Post event) {
        if (writer == null || mc.player == null) return;

        try {
            long timestamp = System.currentTimeMillis();
            double x = mc.player.getX();
            double y = mc.player.getY();
            double z = mc.player.getZ();
            double vx = mc.player.getVelocity().x;
            double vy = mc.player.getVelocity().y;
            double vz = mc.player.getVelocity().z;
            float pitch = mc.player.getPitch();
            float yaw = mc.player.getYaw();

            writer.write(String.format("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                timestamp, x, y, z, vx, vy, vz, pitch, yaw));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
            toggle();
        }
    }
}
