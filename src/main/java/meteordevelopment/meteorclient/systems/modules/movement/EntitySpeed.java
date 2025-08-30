/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.entity.LivingEntityMoveEvent;
import meteordevelopment.meteorclient.mixininterface.IVec3d;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.player.PlayerUtils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.GhastEntity;
import net.minecraft.util.math.Vec3d;

public class EntitySpeed extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> speed = sgGeneral.add(new DoubleSetting.Builder()
        .name("speed")
        .description("Horizontal speed in blocks per second.")
        .defaultValue(10)
        .min(0)
        .sliderMax(50)
        .build()
    );

    private final Setting<Boolean> onlyOnGround = sgGeneral.add(new BoolSetting.Builder()
        .name("only-on-ground")
        .description("Use speed only when standing on a block.")
        .defaultValue(false)
        .build()
    );

    private final Setting<Boolean> inWater = sgGeneral.add(new BoolSetting.Builder()
        .name("in-water")
        .description("Use speed when in water.")
        .defaultValue(false)
        .build()
    );

    public EntitySpeed() {
        super(Categories.Movement, "entity-speed", "Makes you go faster when riding entities.");
    }

    @EventHandler
    private void onLivingEntityMove(LivingEntityMoveEvent event) {
        if (event.entity.getControllingPassenger() != mc.player) return;

        // Check for onlyOnGround and inWater
        LivingEntity entity = event.entity;
        if (onlyOnGround.get() && !entity.isOnGround()) return;
        if (!inWater.get() && entity.isTouchingWater()) return;

        // Set velocity
        Vec3d vel = PlayerUtils.getHorizontalVelocity(speed.get());

        if (entity instanceof GhastEntity) {
            double originalH = Math.sqrt(event.movement.x * event.movement.x + event.movement.z * event.movement.z);
            double newH = Math.sqrt(vel.x * vel.x + vel.z * vel.z);
            double y = event.movement.y;

            if (originalH != 0) y *= newH / originalH;
            else y = 0;

            ((IVec3d) event.movement).meteor$set(vel.x, y, vel.z);
        } else {
            ((IVec3d) event.movement).meteor$setXZ(vel.x, vel.z);
        }
    }
}
