/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.friends.Friends;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.entity.EntityUtils;
import meteordevelopment.meteorclient.utils.entity.SortPriority;
import meteordevelopment.meteorclient.utils.entity.TargetUtils;
import meteordevelopment.meteorclient.utils.player.InvUtils;
import meteordevelopment.meteorclient.utils.player.PlayerUtils;
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.passive.AnimalEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ArrowItem;
import net.minecraft.item.BowItem;
import net.minecraft.item.Items;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

import java.util.Set;

public class LeadBowAimbot extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();
    private final SettingGroup sgRender = settings.createGroup("Render");

    private final Setting<Double> range = sgGeneral.add(new DoubleSetting.Builder()
        .name("range")
        .description("The maximum range the entity can be to aim at it.")
        .defaultValue(20)
        .range(0, 100)
        .sliderMax(100)
        .build()
    );

    private final Setting<Set<EntityType<?>>> entities = sgGeneral.add(new EntityTypeListSetting.Builder()
        .name("entities")
        .description("Entities to target.")
        .onlyAttackable()
        .build()
    );

    private final Setting<Boolean> babies = sgGeneral.add(new BoolSetting.Builder()
        .name("babies")
        .description("Whether or not to attack baby variants of the entity.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Boolean> nametagged = sgGeneral.add(new BoolSetting.Builder()
        .name("nametagged")
        .description("Whether or not to attack mobs with a name tag.")
        .defaultValue(false)
        .build()
    );

    private final Setting<Boolean> render = sgRender.add(new BoolSetting.Builder()
        .name("render")
        .description("Render the predicted hitbox.")
        .defaultValue(true)
        .build()
    );

    private final Setting<ShapeMode> shapeMode = sgRender.add(new EnumSetting.Builder<ShapeMode>()
        .name("shape-mode")
        .description("How the shapes are rendered.")
        .defaultValue(ShapeMode.Lines)
        .build()
    );

    private final Setting<SettingColor> sideColor = sgRender.add(new ColorSetting.Builder()
        .name("side-color")
        .description("The side color.")
        .defaultValue(new SettingColor(45, 255, 115, 25))
        .visible(() -> render.get() && shapeMode.get().sides())
        .build()
    );

    private final Setting<SettingColor> lineColor = sgRender.add(new ColorSetting.Builder()
        .name("line-color")
        .description("The line color.")
        .defaultValue(new SettingColor(45, 255, 115))
        .visible(() -> render.get() && shapeMode.get().lines())
        .build()
    );

    private Entity target;
    private boolean using;

    public LeadBowAimbot() {
        super(Categories.Combat, "lead-bow-aimbot", "Aims bows with lead prediction.");
    }

    @Override
    public void onDeactivate() {
        target = null;
        using = false;
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (!PlayerUtils.isAlive() || !itemInHand()) {
            target = null;
            using = false;
            return;
        }
        if (!mc.player.getAbilities().creativeMode && !InvUtils.find(itemStack -> itemStack.getItem() instanceof ArrowItem).found()) {
            target = null;
            using = false;
            return;
        }

        boolean currentUsing = mc.options.useKey.isPressed() && mc.player.isUsingItem();

        if (currentUsing && !using) {
            target = TargetUtils.get(entity -> {
                if (entity == mc.player || entity == mc.cameraEntity) return false;
                if ((entity instanceof LivingEntity le && le.isDead()) || !entity.isAlive()) return false;
                if (!PlayerUtils.isWithin(entity, range.get())) return false;
                if (!entities.get().contains(entity.getType())) return false;
                if (!nametagged.get() && entity.hasCustomName()) return false;
                if (!PlayerUtils.canSeeEntity(entity)) return false;
                if (entity instanceof PlayerEntity player && !Friends.get().shouldAttack(player)) return false;
                return !(entity instanceof AnimalEntity) || babies.get() || !((AnimalEntity) entity).isBaby();
            }, SortPriority.ClosestAngle);
        }

        using = currentUsing;

        if (target == null) return;
        if (using) {
            aim(event.tickDelta);
            if (render.get()) renderPrediction(event);
        } else {
            target = null;
        }
    }

    private boolean itemInHand() {
        return InvUtils.testInMainHand(Items.BOW);
    }

    private void aim(float tickDelta) {
        float charge = BowItem.getPullProgress(mc.player.getItemUseTime());
        if (charge < 0.1f) return;

        Vec3d eyePos = mc.player.getEyePos();
        Vec3d targetPos = target.getLerpedPos(tickDelta).add(0, target.getHeight() / 2, 0);
        Vec3d targetVel = target.getVelocity();

        double speed = charge * 3.0;
        Vec3d r = targetPos.subtract(eyePos);
        double a = targetVel.lengthSquared() - speed * speed;
        double b = 2 * r.dotProduct(targetVel);
        double c = r.lengthSquared();
        double disc = b * b - 4 * a * c;
        double t;
        if (disc >= 0 && Math.abs(a) > 1e-6) {
            double sqrt = Math.sqrt(disc);
            double t1 = (-b - sqrt) / (2 * a);
            double t2 = (-b + sqrt) / (2 * a);
            t = t1 > 0 ? t1 : t2;
            if (t < 0) t = 0;
        } else if (Math.abs(a) <= 1e-6 && b != 0) {
            t = -c / b;
            if (t < 0) t = 0;
        } else {
            t = Math.sqrt(c) / speed;
        }

        Vec3d predicted = targetPos.add(targetVel.multiply(t));

        double relativeX = predicted.x - mc.player.getX();
        double relativeY = predicted.y - mc.player.getEyeY();
        double relativeZ = predicted.z - mc.player.getZ();

        double hDistance = Math.sqrt(relativeX * relativeX + relativeZ * relativeZ);
        double hDistanceSq = hDistance * hDistance;
        float g = 0.006f;
        float velocitySq = charge * charge;
        float pitch = (float) -Math.toDegrees(Math.atan((velocitySq - Math.sqrt(velocitySq * velocitySq - g * (g * hDistanceSq + 2 * relativeY * velocitySq))) / (g * hDistance)));

        if (Float.isNaN(pitch)) {
            Rotations.rotate(Rotations.getYaw(predicted), Rotations.getPitch(predicted));
        } else {
            Rotations.rotate(Rotations.getYaw(predicted), pitch);
        }
    }

    private void renderPrediction(Render3DEvent event) {
        Vec3d targetPos = target.getLerpedPos(event.tickDelta).add(0, target.getHeight() / 2, 0);
        Vec3d targetVel = target.getVelocity();
        float charge = BowItem.getPullProgress(mc.player.getItemUseTime());
        double speed = charge * 3.0;

        Vec3d eyePos = mc.player.getEyePos();
        Vec3d r = targetPos.subtract(eyePos);
        double a = targetVel.lengthSquared() - speed * speed;
        double b = 2 * r.dotProduct(targetVel);
        double c = r.lengthSquared();
        double disc = b * b - 4 * a * c;
        double t;
        if (disc >= 0 && Math.abs(a) > 1e-6) {
            double sqrt = Math.sqrt(disc);
            double t1 = (-b - sqrt) / (2 * a);
            double t2 = (-b + sqrt) / (2 * a);
            t = t1 > 0 ? t1 : t2;
            if (t < 0) t = 0;
        } else if (Math.abs(a) <= 1e-6 && b != 0) {
            t = -c / b;
            if (t < 0) t = 0;
        } else {
            t = Math.sqrt(c) / speed;
        }

        Vec3d predicted = targetPos.add(targetVel.multiply(t));

        Box box = target.getBoundingBox();
        double dx = predicted.x - target.getX();
        double dy = predicted.y - target.getY();
        double dz = predicted.z - target.getZ();
        event.renderer.box(dx + box.minX, dy + box.minY, dz + box.minZ, dx + box.maxX, dy + box.maxY, dz + box.maxZ, sideColor.get(), lineColor.get(), shapeMode.get(), 0);
    }
}
