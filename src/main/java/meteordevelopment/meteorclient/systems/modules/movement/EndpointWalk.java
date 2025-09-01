/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.util.math.Vec3d;

public class EndpointWalk extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> length = sgGeneral.add(new DoubleSetting.Builder()
        .name("length")
        .description("Distance in blocks from the starting point to each endpoint.")
        .defaultValue(5.0)
        .min(1.0)
        .build()
    );

    private double x1, x2;
    private boolean toFirst;

    public EndpointWalk() {
        super(Categories.Movement, "endpoint-walk", "Faces alternating endpoints along the X axis so you can walk back and forth.");
    }

    @Override
    public void onActivate() {
        double x = mc.player.getX();
        x1 = x + length.get();
        x2 = x - length.get();
        toFirst = true;
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        double targetX = toFirst ? x1 : x2;

        Vec3d target = new Vec3d(targetX, mc.player.getY(), mc.player.getZ());
        float yaw = (float) Rotations.getYaw(target);
        mc.player.setYaw(yaw);
        mc.player.headYaw = yaw;
        mc.player.bodyYaw = yaw;

        if (Math.abs(mc.player.getX() - targetX) <= 1) {
            toFirst = !toFirst;
        }
    }
}

