/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.IntSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.Hand;
import net.minecraft.util.math.Vec3d;

public class FTeleport extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> distance = sgGeneral.add(new DoubleSetting.Builder()
        .name("distance")
        .description("How many blocks forward to teleport.")
        .defaultValue(5)
        .min(0)
        .sliderMax(20)
        .build()
    );

    private final Setting<Integer> delay = sgGeneral.add(new IntSetting.Builder()
        .name("delay")
        .description("How many ticks before teleporting back.")
        .defaultValue(10)
        .min(0)
        .sliderMax(100)
        .build()
    );

    private Vec3d startPos;
    private int timer;

    public FTeleport() {
        super(Categories.Combat, "f-teleport", "Teleports forward then back, attacking players in range.");
    }

    @Override
    public void onActivate() {
        if (mc.player == null || mc.world == null) {
            toggle();
            return;
        }

        startPos = mc.player.getPos();

        Vec3d forward = mc.player.getRotationVec(1f).normalize().multiply(distance.get());
        mc.player.setPosition(startPos.add(forward));

        for (PlayerEntity target : mc.world.getPlayers()) {
            if (target == mc.player) continue;
            if (mc.player.distanceTo(target) <= 6) {
                mc.interactionManager.attackEntity(mc.player, target);
                mc.player.swingHand(Hand.MAIN_HAND);
            }
        }

        timer = delay.get();
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (mc.player == null) return;

        if (timer <= 0) {
            if (startPos != null) mc.player.setPosition(startPos);
            toggle();
            return;
        }

        timer--;
    }

    @Override
    public void onDeactivate() {
        if (mc.player != null && startPos != null) {
            mc.player.setPosition(startPos);
        }
    }
}

