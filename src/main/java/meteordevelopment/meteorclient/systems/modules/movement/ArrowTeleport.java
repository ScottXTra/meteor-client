/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.IntSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.entity.ProjectileEntitySimulator;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.projectile.ArrowEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import org.joml.Vector3d;

public class ArrowTeleport extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> detectDistance = sgGeneral.add(new DoubleSetting.Builder()
        .name("detect-distance")
        .description("How close a projectile trajectory can get before teleporting.")
        .defaultValue(1)
        .min(0.1)
        .sliderRange(0.1, 5)
        .build()
    );

    private final Setting<Double> maxTeleport = sgGeneral.add(new DoubleSetting.Builder()
        .name("max-teleport")
        .description("Maximum teleport distance.")
        .defaultValue(4)
        .min(1)
        .sliderRange(1, 8)
        .build()
    );

    private final Setting<Boolean> groundCheck = sgGeneral.add(new BoolSetting.Builder()
        .name("ground-check")
        .description("Don't teleport into air when enabled.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Integer> simulationSteps = sgGeneral.add(new IntSetting.Builder()
        .name("simulation-steps")
        .description("How many ticks ahead to simulate arrows.")
        .defaultValue(80)
        .sliderRange(10, 200)
        .build()
    );

    private final ProjectileEntitySimulator simulator = new ProjectileEntitySimulator();
    private final Vector3d vec = new Vector3d();

    public ArrowTeleport() {
        super(Categories.Movement, "arrow-teleport", "Teleports perpendicular to incoming arrows.");
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        for (Entity e : mc.world.getEntities()) {
            if (!(e instanceof ArrowEntity)) continue;
            if (!simulator.set(e, true)) continue;

            Vec3d playerPos = mc.player.getPos();
            Vec3d arrowVel = e.getVelocity();
            Vec3d norm = arrowVel.normalize();

            vec.set(simulator.pos);
            for (int i = 0; i < simulationSteps.get(); i++) {
                Vec3d pos = new Vec3d(vec.x, vec.y, vec.z);
                double t = playerPos.subtract(pos).dotProduct(norm);
                if (t >= 0) {
                    Vec3d closest = pos.add(norm.multiply(t));
                    if (closest.isInRange(playerPos, detectDistance.get())) {
                        teleportPerpendicular(norm);
                        return;
                    }
                }
                if (simulator.tick() != null) break;
                vec.set(simulator.pos);
            }
        }
    }

    private void teleportPerpendicular(Vec3d direction) {
        Vec3d perp = new Vec3d(-direction.z, 0, direction.x).normalize();
        Vec3d negPerp = perp.multiply(-1, 1, -1);
        Vec3d playerPos = mc.player.getPos();

        double step = 0.5;
        for (double dist = step; dist <= maxTeleport.get(); dist += step) {
            Vec3d target = playerPos.add(perp.multiply(dist));
            if (isSafe(target)) {
                mc.player.setPosition(target.x, target.y, target.z);
                return;
            }
            target = playerPos.add(negPerp.multiply(dist));
            if (isSafe(target)) {
                mc.player.setPosition(target.x, target.y, target.z);
                return;
            }
        }
    }

    private boolean isSafe(Vec3d pos) {
        BlockPos bp = BlockPos.ofFloored(pos);
        if (!mc.world.getBlockState(bp).getCollisionShape(mc.world, bp).isEmpty()) return false;
        if (!mc.world.getBlockState(bp.up()).getCollisionShape(mc.world, bp.up()).isEmpty()) return false;
        if (groundCheck.get()) {
            return !mc.world.getBlockState(bp.down()).getCollisionShape(mc.world, bp.down()).isEmpty();
        }
        return true;
    }
}
