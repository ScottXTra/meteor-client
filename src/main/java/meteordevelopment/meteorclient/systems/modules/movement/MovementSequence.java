/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.IntSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.orbit.EventHandler;

public class MovementSequence extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Integer> yawDelay = sgGeneral.add(new IntSetting.Builder()
        .name("yaw-delay")
        .description("Ticks to wait after aligning yaw.")
        .defaultValue(2)
        .min(0)
        .build()
    );

    private final Setting<Integer> sprintTicks = sgGeneral.add(new IntSetting.Builder()
        .name("sprint-ticks")
        .description("Ticks to sprint forward.")
        .defaultValue(5)
        .min(0)
        .build()
    );

    private final Setting<Integer> waitAfterSprint = sgGeneral.add(new IntSetting.Builder()
        .name("wait-after-sprint")
        .description("Ticks to wait after sprinting.")
        .defaultValue(10)
        .min(0)
        .build()
    );

    private final Setting<Integer> waitAfterCrouch = sgGeneral.add(new IntSetting.Builder()
        .name("wait-after-crouch")
        .description("Ticks to wait after crouching.")
        .defaultValue(10)
        .min(0)
        .build()
    );

    private final Setting<Integer> backTicks = sgGeneral.add(new IntSetting.Builder()
        .name("back-ticks")
        .description("Ticks to move backwards while crouching.")
        .defaultValue(1)
        .min(0)
        .build()
    );

    private final Setting<Integer> waitAfterBack = sgGeneral.add(new IntSetting.Builder()
        .name("wait-after-back")
        .description("Ticks to wait after moving backwards.")
        .defaultValue(10)
        .min(0)
        .build()
    );

    private final Setting<Integer> waitAfterYawAdjust = sgGeneral.add(new IntSetting.Builder()
        .name("wait-after-yaw-adjust")
        .description("Ticks to wait after adjusting yaw.")
        .defaultValue(10)
        .min(0)
        .build()
    );

    private final Setting<Integer> forwardTicks = sgGeneral.add(new IntSetting.Builder()
        .name("forward-ticks")
        .description("Ticks to move forward while crouching.")
        .defaultValue(1)
        .min(0)
        .build()
    );

    private final Setting<Integer> finalWait = sgGeneral.add(new IntSetting.Builder()
        .name("final-wait")
        .description("Ticks to wait before disabling.")
        .defaultValue(10)
        .min(0)
        .build()
    );

    private int step;
    private int timer;

    public MovementSequence() {
        super(Categories.Movement, "movement-sequence", "Performs a scripted sequence of movements.");
    }

    @Override
    public void onActivate() {
        if (mc.player == null) {
            toggle();
            return;
        }

        step = 0;
        timer = 0;

        float yaw = mc.player.getYaw();
        float target = Math.round(yaw / 90f) * 90f;
        mc.player.setYaw(target);
        mc.player.headYaw = target;
        mc.player.bodyYaw = target;
    }

    @Override
    public void onDeactivate() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.sprintKey.setPressed(false);
        mc.options.sneakKey.setPressed(false);
        if (mc.player != null) mc.player.setSprinting(false);
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        if (mc.player == null) {
            toggle();
            return;
        }

        switch (step) {
            case 0 -> {
                if (timer++ >= yawDelay.get()) {
                    timer = 0;
                    step = 1;
                }
            }
            case 1 -> {
                mc.options.forwardKey.setPressed(true);
                mc.player.setSprinting(true);
                if (timer++ >= sprintTicks.get()) {
                    mc.options.forwardKey.setPressed(false);
                    mc.player.setSprinting(false);
                    timer = 0;
                    step = 2;
                }
            }
            case 2 -> {
                if (timer++ >= waitAfterSprint.get()) {
                    mc.options.sneakKey.setPressed(true);
                    timer = 0;
                    step = 3;
                }
            }
            case 3 -> {
                if (timer++ >= waitAfterCrouch.get()) {
                    timer = 0;
                    step = 4;
                }
            }
            case 4 -> {
                mc.options.backKey.setPressed(true);
                if (timer++ >= backTicks.get()) {
                    mc.options.backKey.setPressed(false);
                    timer = 0;
                    step = 5;
                }
            }
            case 5 -> {
                if (timer++ >= waitAfterBack.get()) {
                    float yaw = mc.player.getYaw() + 0.1f;
                    mc.player.setYaw(yaw);
                    mc.player.headYaw = yaw;
                    mc.player.bodyYaw = yaw;
                    timer = 0;
                    step = 6;
                }
            }
            case 6 -> {
                if (timer++ >= waitAfterYawAdjust.get()) {
                    timer = 0;
                    step = 7;
                }
            }
            case 7 -> {
                mc.options.forwardKey.setPressed(true);
                if (timer++ >= forwardTicks.get()) {
                    mc.options.forwardKey.setPressed(false);
                    timer = 0;
                    step = 8;
                }
            }
            case 8 -> {
                if (timer++ >= finalWait.get()) {
                    float yaw = mc.player.getYaw();
                    float target = Math.round(yaw / 90f) * 90f;
                    mc.player.setYaw(target);
                    mc.player.headYaw = target;
                    mc.player.bodyYaw = target;
                    timer = 0;
                    step = 9;
                }
            }
            case 9 -> {
                if (timer++ >= yawDelay.get()) {
                    toggle();
                }
            }
        }
    }
}

