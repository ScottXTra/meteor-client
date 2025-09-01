/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.projectile.ProjectileUtil;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Vec3d;

public class TeleportHit extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private Vec3d serverTeleportPos = Vec3d.ZERO;
    private int renderTicks;

    private final Setting<Double> maxDistance = sgGeneral.add(new DoubleSetting.Builder()
        .name("max-distance")
        .description("Max teleport distance.")
        .defaultValue(100.0)
        .min(0.0)
        .sliderMax(300.0)
        .build()
    );

    public TeleportHit() {
        super(Categories.Combat, "teleport-hit", "Allows you to attack enemies from further distances.");
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (renderTicks > 0) renderTicks--;

        if (mc.player == null || mc.world == null) return;
        if (!mc.options.attackKey.isPressed()) return;

        Vec3d start = mc.player.getEyePos();
        Vec3d direction = mc.player.getRotationVec(1f);
        Vec3d end = start.add(direction.multiply(maxDistance.get()));
        Box box = mc.player.getBoundingBox().stretch(direction.multiply(maxDistance.get())).expand(1.0);

        EntityHitResult hit = ProjectileUtil.getEntityCollision(
            mc.world,
            mc.player,
            start,
            end,
            box,
            entity -> !entity.isSpectator() && entity.canHit(),
            ProjectileUtil.getToleranceMargin(mc.player)
        );

        if (hit == null) return;
        Entity target = hit.getEntity();

        Vec3d startPos = mc.player.getPos();
        Vec3d targetPos = target.getPos();

        teleport(targetPos);
        mc.interactionManager.attackEntity(mc.player, target);
        mc.player.swingHand(Hand.MAIN_HAND);
        teleport(startPos);

        serverTeleportPos = targetPos;
        renderTicks = 2;
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (renderTicks <= 0 || mc.player == null) return;

        Box box = mc.player.getBoundingBox().offset(
            serverTeleportPos.x - mc.player.getX(),
            serverTeleportPos.y - mc.player.getY(),
            serverTeleportPos.z - mc.player.getZ()
        );

        event.renderer.box(box, Color.WHITE, Color.WHITE, ShapeMode.Lines, 0);
        event.renderer.line(mc.player.getX(), mc.player.getY(), mc.player.getZ(),
            serverTeleportPos.x, serverTeleportPos.y, serverTeleportPos.z, Color.WHITE);
    }

    private void teleport(Vec3d pos) {
        mc.player.setPosition(pos.x, pos.y, pos.z);
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.PositionAndOnGround(pos.x, pos.y, pos.z, mc.player.isOnGround(), mc.player.horizontalCollision));
    }
}

