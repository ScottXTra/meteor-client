/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.combat;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
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
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

import java.util.*;

public class TeleportHit extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private int renderTicks;
    private List<Vec3d> renderPath = Collections.emptyList();

    private final Setting<Double> maxDistance = sgGeneral.add(new DoubleSetting.Builder()
        .name("max-distance")
        .description("Max teleport distance.")
        .defaultValue(100.0)
        .min(0.0)
        .sliderMax(300.0)
        .build()
    );

    private final Setting<Double> maxTeleportDistance = sgGeneral.add(new DoubleSetting.Builder()
        .name("max-teleport-distance")
        .description("Maximum distance per teleport step.")
        .defaultValue(5.0)
        .min(1.0)
        .max(7.0)
        .sliderMax(7.0)
        .build()
    );

    private final Setting<Boolean> multiTeleport = sgGeneral.add(new BoolSetting.Builder()
        .name("multi-teleport")
        .description("Teleport in multiple steps.")
        .defaultValue(true)
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

        List<Vec3d> path = findPath(startPos, targetPos);
        if (path == null || path.size() < 2) return;

        List<Vec3d> teleports = new ArrayList<>();

        if (multiTeleport.get()) {
            teleports = teleportAlong(path);
            mc.interactionManager.attackEntity(mc.player, target);
            mc.player.swingHand(Hand.MAIN_HAND);
            teleportBack(teleports, startPos);
        } else {
            Vec3d pos = path.get(path.size() - 1);
            teleport(pos);
            teleports.add(pos);
            mc.interactionManager.attackEntity(mc.player, target);
            mc.player.swingHand(Hand.MAIN_HAND);
            teleport(startPos);
        }

        renderPath = new ArrayList<>();
        renderPath.add(startPos);
        renderPath.addAll(teleports);
        renderTicks = 2;
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        if (renderTicks <= 0 || mc.player == null) return;
        if (renderPath == null || renderPath.size() < 2) return;

        if (multiTeleport.get()) {
            for (int i = 1; i < renderPath.size(); i++) {
                Vec3d pos = renderPath.get(i);
                Box box = mc.player.getBoundingBox().offset(
                    pos.x - mc.player.getX(),
                    pos.y - mc.player.getY(),
                    pos.z - mc.player.getZ()
                );
                event.renderer.box(box, Color.WHITE, Color.WHITE, ShapeMode.Lines, 0);
            }

            for (int i = 0; i < renderPath.size() - 1; i++) {
                Vec3d a = renderPath.get(i);
                Vec3d b = renderPath.get(i + 1);
                event.renderer.line(a.x, a.y, a.z, b.x, b.y, b.z, Color.WHITE);
            }
        } else {
            Vec3d pos = renderPath.get(renderPath.size() - 1);
            Box box = mc.player.getBoundingBox().offset(
                pos.x - mc.player.getX(),
                pos.y - mc.player.getY(),
                pos.z - mc.player.getZ()
            );
            event.renderer.box(box, Color.WHITE, Color.WHITE, ShapeMode.Lines, 0);
            event.renderer.line(mc.player.getX(), mc.player.getY(), mc.player.getZ(),
                pos.x, pos.y, pos.z, Color.WHITE);
        }
    }

    private List<Vec3d> findPath(Vec3d start, Vec3d end) {
        BlockPos startPos = BlockPos.ofFloored(start);
        BlockPos endPos = BlockPos.ofFloored(end);
        List<BlockPos> blocks = aStar(startPos, endPos);
        if (blocks == null) return null;

        List<Vec3d> path = new ArrayList<>(blocks.size());
        for (BlockPos pos : blocks) path.add(Vec3d.ofCenter(pos));
        return path;
    }

    private List<BlockPos> aStar(BlockPos start, BlockPos goal) {
        PriorityQueue<Node> open = new PriorityQueue<>(Comparator.comparingDouble(n -> n.f));
        Map<BlockPos, Node> all = new HashMap<>();

        Node startNode = new Node(start, null, 0, start.getManhattanDistance(goal));
        open.add(startNode);
        all.put(start, startNode);

        while (!open.isEmpty()) {
            Node current = open.poll();
            if (current.pos.equals(goal)) {
                List<BlockPos> path = new ArrayList<>();
                while (current != null) {
                    path.add(0, current.pos);
                    current = current.prev;
                }
                return path;
            }

            for (Direction dir : Direction.values()) {
                BlockPos neighborPos = current.pos.offset(dir);
                if (!isSafe(neighborPos)) continue;

                double g = current.g + 1;
                Node neighbor = all.get(neighborPos);
                double h = neighborPos.getManhattanDistance(goal);
                if (neighbor == null || g < neighbor.g) {
                    if (neighbor == null) {
                        neighbor = new Node(neighborPos, current, g, g + h);
                        all.put(neighborPos, neighbor);
                        open.add(neighbor);
                    } else {
                        neighbor.prev = current;
                        neighbor.g = g;
                        neighbor.f = g + h;
                        open.remove(neighbor);
                        open.add(neighbor);
                    }
                }
            }
        }

        return null;
    }

    private boolean isSafe(BlockPos pos) {
        return mc.world.getBlockState(pos).isAir() && mc.world.getBlockState(pos.up()).isAir();
    }

    private List<Vec3d> teleportAlong(List<Vec3d> path) {
        List<Vec3d> teleports = new ArrayList<>();
        int step = Math.max(1, (int) Math.floor(maxTeleportDistance.get()));
        for (int i = step; i < path.size(); i += step) {
            Vec3d pos = path.get(i);
            teleport(pos);
            teleports.add(pos);
        }
        if ((path.size() - 1) % step != 0) {
            Vec3d pos = path.get(path.size() - 1);
            teleport(pos);
            teleports.add(pos);
        }
        return teleports;
    }

    private void teleportBack(List<Vec3d> teleports, Vec3d startPos) {
        for (int i = teleports.size() - 2; i >= 0; i--) teleport(teleports.get(i));
        teleport(startPos);
    }

    private static class Node {
        final BlockPos pos;
        Node prev;
        double g;
        double f;

        Node(BlockPos pos, Node prev, double g, double f) {
            this.pos = pos;
            this.prev = prev;
            this.g = g;
            this.f = f;
        }
    }

    private void teleport(Vec3d pos) {
        mc.player.setPosition(pos.x, pos.y, pos.z);
        mc.player.networkHandler.sendPacket(new PlayerMoveC2SPacket.PositionAndOnGround(pos.x, pos.y, pos.z, mc.player.isOnGround(), mc.player.horizontalCollision));
    }
}

