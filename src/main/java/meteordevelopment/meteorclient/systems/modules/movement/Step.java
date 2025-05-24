/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import com.google.common.collect.Streams;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.pathing.PathManagers;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.entity.DamageUtils;
import meteordevelopment.meteorclient.utils.player.FindItemResult;
import meteordevelopment.meteorclient.utils.player.InvUtils;
import meteordevelopment.meteorclient.utils.world.BlockUtils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.attribute.EntityAttributes;
import net.minecraft.entity.decoration.EndCrystalEntity;
import net.minecraft.entity.vehicle.AbstractBoatEntity;
import net.minecraft.item.BoatItem;
import net.minecraft.util.Hand;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

import java.util.OptionalDouble;

public class Step extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    public final Setting<Double> height = sgGeneral.add(new DoubleSetting.Builder()
        .name("height")
        .description("Step height.")
        .defaultValue(1)
        .min(0)
        .build()
    );

    private final Setting<ActiveWhen> activeWhen = sgGeneral.add(new EnumSetting.Builder<ActiveWhen>()
        .name("active-when")
        .description("Step is active when you meet these requirements.")
        .defaultValue(ActiveWhen.Always)
        .build()
    );

    private final Setting<Boolean> safeStep = sgGeneral.add(new BoolSetting.Builder()
        .name("safe-step")
        .description("Doesn't let you step out of a hole if you are low on health or there is a crystal nearby.")
        .defaultValue(false)
        .build()
    );

    private final Setting<Integer> stepHealth = sgGeneral.add(new IntSetting.Builder()
        .name("step-health")
        .description("The health you stop being able to step at.")
        .defaultValue(5)
        .range(1, 36)
        .sliderRange(1, 36)
        .visible(safeStep::get)
        .build()
    );

    private final Setting<Boolean> boatStep = sgGeneral.add(new BoolSetting.Builder()
        .name("boat-step")
        .description("Places a boat before stepping and breaks it afterwards.")
        .defaultValue(false)
        .build()
    );

    private float prevStepHeight;
    private boolean prevPathManagerStep;
    private AbstractBoatEntity placedBoat;
    private double lastY;

    public Step() {
        super(Categories.Movement, "step", "Allows you to walk up full blocks instantly.");
    }

    @Override
    public void onActivate() {
        prevStepHeight = mc.player.getStepHeight();

        placedBoat = null;
        lastY = mc.player.getY();

        prevPathManagerStep = PathManagers.get().getSettings().getStep().get();
        PathManagers.get().getSettings().getStep().set(true);
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        boolean work = (activeWhen.get() == ActiveWhen.Always) || (activeWhen.get() == ActiveWhen.Sneaking && mc.player.isSneaking()) || (activeWhen.get() == ActiveWhen.NotSneaking && !mc.player.isSneaking());

        if (boatStep.get()) {
            if (placedBoat != null) {
                if (mc.player.getY() > lastY + 0.1) {
                    mc.interactionManager.attackEntity(mc.player, placedBoat);
                    mc.player.swingHand(Hand.MAIN_HAND);
                    placedBoat = null;
                }
            } else if (mc.player.horizontalCollision && mc.player.isOnGround() && work) {
                BlockPos pos = mc.player.getBlockPos().offset(Direction.fromRotation(mc.player.getYaw()));
                FindItemResult boat = InvUtils.findInHotbar(item -> item instanceof BoatItem);
                if (boat.found()) {
                    if (BlockUtils.place(pos, boat, true, 50)) {
                        placedBoat = findBoatAt(pos);
                    }
                }
            }
        }

        mc.player.setBoundingBox(mc.player.getBoundingBox().offset(0, 1, 0));
        if (work && (!safeStep.get() || (getHealth() > stepHealth.get() && getHealth() - getExplosionDamage() > stepHealth.get()))){
            mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(height.get());
        } else {
            mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(prevStepHeight);
        }
        mc.player.setBoundingBox(mc.player.getBoundingBox().offset(0, -1, 0));

        lastY = mc.player.getY();
    }

    @Override
    public void onDeactivate() {
        mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(prevStepHeight);

        if (placedBoat != null) {
            mc.interactionManager.attackEntity(mc.player, placedBoat);
            mc.player.swingHand(Hand.MAIN_HAND);
            placedBoat = null;
        }

        PathManagers.get().getSettings().getStep().set(prevPathManagerStep);
    }

    private float getHealth() {
        return mc.player.getHealth() + mc.player.getAbsorptionAmount();
    }

    private double getExplosionDamage() {
        OptionalDouble crystalDamage = Streams.stream(mc.world.getEntities())
                .filter(entity -> entity instanceof EndCrystalEntity)
                .filter(Entity::isAlive)
                .mapToDouble(entity -> DamageUtils.crystalDamage(mc.player, entity.getPos()))
                .max();
        return crystalDamage.orElse(0.0);
    }

    private AbstractBoatEntity findBoatAt(BlockPos pos) {
        AbstractBoatEntity best = null;
        double bestDistance = Double.MAX_VALUE;
        for (Entity entity : mc.world.getEntities()) {
            if (entity instanceof AbstractBoatEntity boat) {
                double d = boat.squaredDistanceTo(Vec3d.ofCenter(pos));
                if (d < bestDistance) {
                    bestDistance = d;
                    best = boat;
                }
            }
        }
        return best;
    }

    public enum ActiveWhen {
        Always,
        Sneaking,
        NotSneaking
    }
}
