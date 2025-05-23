/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.render;

import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.Utils;
import meteordevelopment.meteorclient.utils.entity.ProjectileEntitySimulator;
import meteordevelopment.meteorclient.utils.misc.Pool;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.projectile.ProjectileEntity;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.MathHelper;
import org.joml.Vector3d;

import java.util.ArrayList;
import java.util.List;

public class ProjectileTrajectories extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();
    private final SettingGroup sgRender = settings.createGroup("Render");

    // General

    private final Setting<Boolean> accurate = sgGeneral.add(new BoolSetting.Builder()
        .name("accurate")
        .description("Whether or not to calculate more accurate.")
        .defaultValue(false)
        .build()
    );

    public final Setting<Integer> simulationSteps = sgGeneral.add(new IntSetting.Builder()
        .name("simulation-steps")
        .description("How many steps to simulate projectiles. Zero for no limit")
        .defaultValue(500)
        .sliderMax(5000)
        .build()
    );

    // Render

    private final Setting<ShapeMode> shapeMode = sgRender.add(new EnumSetting.Builder<ShapeMode>()
        .name("shape-mode")
        .description("How the shapes are rendered.")
        .defaultValue(ShapeMode.Both)
        .build()
    );

    private final Setting<SettingColor> sideColor = sgRender.add(new ColorSetting.Builder()
        .name("side-color")
        .description("The side color.")
        .defaultValue(new SettingColor(255, 150, 0, 35))
        .build()
    );

    private final Setting<SettingColor> lineColor = sgRender.add(new ColorSetting.Builder()
        .name("line-color")
        .description("The line color.")
        .defaultValue(new SettingColor(255, 150, 0))
        .build()
    );

    private final Setting<Boolean> renderPositionBox = sgRender.add(new BoolSetting.Builder()
        .name("render-position-boxes")
        .description("Renders the actual position the projectile will be at each tick along it's trajectory.")
        .defaultValue(false)
        .build()
    );

    private final Setting<Double> positionBoxSize = sgRender.add(new DoubleSetting.Builder()
        .name("position-box-size")
        .description("The size of the box drawn at the simulated positions.")
        .defaultValue(0.02)
        .sliderRange(0.01, 0.1)
        .visible(renderPositionBox::get)
        .build()
    );

    private final Setting<SettingColor> positionSideColor = sgRender.add(new ColorSetting.Builder()
        .name("position-side-color")
        .description("The side color.")
        .defaultValue(new SettingColor(255, 150, 0, 35))
        .visible(renderPositionBox::get)
        .build()
    );

    private final Setting<SettingColor> positionLineColor = sgRender.add(new ColorSetting.Builder()
        .name("position-line-color")
        .description("The line color.")
        .defaultValue(new SettingColor(255, 150, 0))
        .visible(renderPositionBox::get)
        .build()
    );

    private final ProjectileEntitySimulator simulator = new ProjectileEntitySimulator();

    private final Pool<Vector3d> vec3s = new Pool<>(Vector3d::new);
    private final List<Path> paths = new ArrayList<>();

    public ProjectileTrajectories() {
        super(Categories.Render, "projectile-trajectories", "Predicts the trajectory of fired projectiles.");
    }

    private Path getEmptyPath() {
        for (Path path : paths) {
            if (path.points.isEmpty()) return path;
        }

        Path path = new Path();
        paths.add(path);
        return path;
    }

    private void calculateFiredPath(Entity entity, double tickDelta) {
        for (Path path : paths) path.clear();

        if (!simulator.set(entity, accurate.get())) return;
        getEmptyPath().setStart(entity, tickDelta).calculate();
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        float tickDelta = mc.world.getTickManager().isFrozen() ? 1 : event.tickDelta;

        for (Entity entity : mc.world.getEntities()) {
            if (entity instanceof ProjectileEntity) {
                calculateFiredPath(entity, tickDelta);
                for (Path path : paths) path.render(event);
            }
        }
    }

    private class Path {
        private final List<Vector3d> points = new ArrayList<>();

        private boolean hitQuad, hitQuadHorizontal;
        private double hitQuadX1, hitQuadY1, hitQuadZ1, hitQuadX2, hitQuadY2, hitQuadZ2;

        private Entity collidingEntity;
        public Vector3d lastPoint;

        public void clear() {
            for (Vector3d point : points) vec3s.free(point);
            points.clear();

            hitQuad = false;
            collidingEntity = null;
            lastPoint = null;
        }

        public void calculate() {
            addPoint();

            for (int i = 0; i < (simulationSteps.get() > 0 ? simulationSteps.get() : Integer.MAX_VALUE); i++) {
                HitResult result = simulator.tick();

                if (result != null) {
                    processHitResult(result);
                    break;
                }

                addPoint();
            }
        }

        public Path setStart(Entity entity, double tickDelta) {
            lastPoint = new Vector3d(
                MathHelper.lerp(tickDelta, entity.lastRenderX, entity.getX()),
                MathHelper.lerp(tickDelta, entity.lastRenderY, entity.getY()),
                MathHelper.lerp(tickDelta, entity.lastRenderZ, entity.getZ())
            );

            return this;
        }

        private void addPoint() {
            points.add(vec3s.get().set(simulator.pos));
        }

        private void processHitResult(HitResult result) {
            if (result.getType() == HitResult.Type.BLOCK) {
                BlockHitResult r = (BlockHitResult) result;

                hitQuad = true;
                hitQuadX1 = r.getPos().x;
                hitQuadY1 = r.getPos().y;
                hitQuadZ1 = r.getPos().z;
                hitQuadX2 = r.getPos().x;
                hitQuadY2 = r.getPos().y;
                hitQuadZ2 = r.getPos().z;

                if (r.getSide() == Direction.UP || r.getSide() == Direction.DOWN) {
                    hitQuadHorizontal = true;
                    hitQuadX1 -= 0.25;
                    hitQuadZ1 -= 0.25;
                    hitQuadX2 += 0.25;
                    hitQuadZ2 += 0.25;
                }
                else if (r.getSide() == Direction.NORTH || r.getSide() == Direction.SOUTH) {
                    hitQuadHorizontal = false;
                    hitQuadX1 -= 0.25;
                    hitQuadY1 -= 0.25;
                    hitQuadX2 += 0.25;
                    hitQuadY2 += 0.25;
                }
                else {
                    hitQuadHorizontal = false;
                    hitQuadZ1 -= 0.25;
                    hitQuadY1 -= 0.25;
                    hitQuadZ2 += 0.25;
                    hitQuadY2 += 0.25;
                }

                points.add(Utils.set(vec3s.get(), result.getPos()));
            }
            else if (result.getType() == HitResult.Type.ENTITY) {
                collidingEntity = ((EntityHitResult) result).getEntity();

                points.add(Utils.set(vec3s.get(), result.getPos()).add(0, collidingEntity.getHeight() / 2, 0));
            }
        }

        public void render(Render3DEvent event) {
            // Render path
            for (Vector3d point : points) {
                if (lastPoint != null) {
                    event.renderer.line(lastPoint.x, lastPoint.y, lastPoint.z, point.x, point.y, point.z, lineColor.get());
                    if (renderPositionBox.get())
                        event.renderer.box(point.x - positionBoxSize.get(), point.y - positionBoxSize.get(), point.z - positionBoxSize.get(),
                            point.x + positionBoxSize.get(), point.y + positionBoxSize.get(), point.z + positionBoxSize.get(), positionSideColor.get(), positionLineColor.get(), shapeMode.get(), 0);
                }
                lastPoint = point;
            }

            // Render hit quad
            if (hitQuad) {
                if (hitQuadHorizontal) event.renderer.sideHorizontal(hitQuadX1, hitQuadY1, hitQuadZ1, hitQuadX1 + 0.5, hitQuadZ1 + 0.5, sideColor.get(), lineColor.get(), shapeMode.get());
                else event.renderer.sideVertical(hitQuadX1, hitQuadY1, hitQuadZ1, hitQuadX2, hitQuadY2, hitQuadZ2, sideColor.get(), lineColor.get(), shapeMode.get());
            }

            // Render entity
            if (collidingEntity != null) {
                double x = (collidingEntity.getX() - collidingEntity.lastX) * event.tickDelta;
                double y = (collidingEntity.getY() - collidingEntity.lastY) * event.tickDelta;
                double z = (collidingEntity.getZ() - collidingEntity.lastZ) * event.tickDelta;

                Box box = collidingEntity.getBoundingBox();
                event.renderer.box(x + box.minX, y + box.minY, z + box.minZ, x + box.maxX, y + box.maxY, z + box.maxZ, sideColor.get(), lineColor.get(), shapeMode.get(), 0);
            }
        }
    }
}
