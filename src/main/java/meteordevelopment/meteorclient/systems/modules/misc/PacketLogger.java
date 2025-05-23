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
import net.minecraft.network.packet.s2c.play.PlayerPositionLookS2CPacket;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Logs sent packets and server position corrections to a file.
 */
public class PacketLogger extends Module {
    private BufferedWriter writer;
    private File file;

    public PacketLogger() {
        super(Categories.Misc, "packet-logger", "Logs sent packets and server rubberbands.");
    }

    @Override
    public void onActivate() {
        try {
            file = new File(MeteorClient.FOLDER, "packet-log.txt");
            file.getParentFile().mkdirs();
            writer = new BufferedWriter(new FileWriter(file, false));
        } catch (IOException e) {
            error("Failed to open log file");
            e.printStackTrace();
            toggle();
        }
    }

    @Override
    public void onDeactivate() {
        try {
            if (writer != null) writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void log(String msg) {
        if (writer == null) return;
        try {
            writer.write(msg);
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
            toggle();
        }
    }

    @EventHandler
    private void onSendPacket(PacketEvent.Send event) {
        Packet<?> packet = event.packet;
        @SuppressWarnings("unchecked")
        String name = PacketUtils.getName((Class<? extends Packet<?>>) packet.getClass());
        if (name == null) name = packet.getClass().getSimpleName();
        log("SEND " + name);
    }

    @EventHandler
    private void onReceivePacket(PacketEvent.Receive event) {
        if (event.packet instanceof PlayerPositionLookS2CPacket packet) {
            log("SERVER_POS " + packet.change().position());
        }
    }
}
