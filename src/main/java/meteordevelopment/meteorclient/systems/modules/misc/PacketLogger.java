/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.network.PacketUtils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.network.packet.Packet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Logs all outgoing packets to a file and chat.
 */
public class PacketLogger extends Module {
    private BufferedWriter writer;

    public PacketLogger() {
        super(Categories.Misc, "packet-logger", "Logs all outgoing packets to a file and chat.");
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

        String name = PacketUtils.getName(packetClass);
        if (name == null) name = packetClass.getName();

        info(name);

        if (writer != null) {
            try {
                writer.write(name);
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
