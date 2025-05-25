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
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.attribute.EntityAttributes;
import net.minecraft.entity.decoration.EndCrystalEntity;
import net.minecraft.entity.vehicle.BoatEntity;
import net.minecraft.entity.vehicle.ChestBoatEntity;
import net.minecraft.item.BoatItem;
import net.minecraft.network.packet.c2s.play.HandSwingC2SPacket;
import net.minecraft.network.packet.c2s.play.PlayerInteractEntityC2SPacket;
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
        .description("Automatically places a boat before stepping and breaks it after stepping.")
        .defaultValue(false)
        .build()
    );

    private Entity placedBoat;
    private int boatTimer;

    private float prevStepHeight;
    private boolean prevPathManagerStep;

    public Step() {
        super(Categories.Movement, "step", "Allows you to walk up full blocks instantly.");
    }

    @Override
    public void onActivate() {
        prevStepHeight = mc.player.getStepHeight();

        prevPathManagerStep = PathManagers.get().getSettings().getStep().get();
        PathManagers.get().getSettings().getStep().set(true);

        placedBoat = null;
        boatTimer = 0;
    }

    @EventHandler
    private void onTick(TickEvent.Post event) {
        boolean work = (activeWhen.get() == ActiveWhen.Always) || (activeWhen.get() == ActiveWhen.Sneaking && mc.player.isSneaking()) || (activeWhen.get() == ActiveWhen.NotSneaking && !mc.player.isSneaking());
        mc.player.setBoundingBox(mc.player.getBoundingBox().offset(0, 1, 0));
        if (work && (!safeStep.get() || (getHealth() > stepHealth.get() && getHealth() - getExplosionDamage() > stepHealth.get()))){
            mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(height.get());
            if (boatStep.get()) handleBoatStep();
        } else {
            mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(prevStepHeight);
        }
        mc.player.setBoundingBox(mc.player.getBoundingBox().offset(0, -1, 0));
    }

    @Override
    public void onDeactivate() {
        mc.player.getAttributeInstance(EntityAttributes.STEP_HEIGHT).setBaseValue(prevStepHeight);

        PathManagers.get().getSettings().getStep().set(prevPathManagerStep);

        placedBoat = null;
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

    private void handleBoatStep() {
        if (placedBoat != null) {
            boatTimer++;

            if (!placedBoat.isAlive()) {
                placedBoat = null;
                return;
            }

            if (boatTimer > 1) {
                mc.player.networkHandler.sendPacket(PlayerInteractEntityC2SPacket.attack(placedBoat, mc.player.isSneaking()));
                mc.getNetworkHandler().sendPacket(new HandSwingC2SPacket(Hand.MAIN_HAND));
                placedBoat = null;
            }

            return;
        }

        Direction dir = mc.player.getHorizontalFacing();
        BlockPos pos = mc.player.getBlockPos().offset(dir);
        if (!mc.world.getBlockState(pos).isAir() && mc.world.getBlockState(pos.up()).isAir()) {
            FindItemResult boat = InvUtils.findInHotbar(stack -> stack.getItem() instanceof BoatItem);
            if (boat.found()) {
                Vec3d hitPos = Vec3d.ofCenter(pos.up());
                Rotations.rotate(Rotations.getYaw(hitPos), Rotations.getPitch(hitPos), () -> {
                    if (boat.getHand() != null) mc.interactionManager.interactItem(mc.player, boat.getHand());
                    else {
                        InvUtils.swap(boat.slot(), true);
                        mc.interactionManager.interactItem(mc.player, Hand.MAIN_HAND);
                        InvUtils.swapBack();
                    }
                });

                for (Entity entity : mc.world.getEntities()) {
                    if ((entity instanceof BoatEntity || entity instanceof ChestBoatEntity) && entity.squaredDistanceTo(hitPos) < 4) {
                        placedBoat = entity;
                        boatTimer = 0;
                        break;
                    }
                }
            }
        }
    }

    public enum ActiveWhen {
        Always,
        Sneaking,
        NotSneaking
    }
}
