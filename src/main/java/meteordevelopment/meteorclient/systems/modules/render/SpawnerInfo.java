/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.render;

import meteordevelopment.meteorclient.events.render.Render2DEvent;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.Utils;
import meteordevelopment.meteorclient.utils.player.PlayerUtils;
import meteordevelopment.meteorclient.utils.render.NametagUtils;
import meteordevelopment.meteorclient.renderer.text.TextRenderer;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.block.entity.MobSpawnerBlockEntity;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import org.joml.Vector3d;

public class SpawnerInfo extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> scale = sgGeneral.add(new DoubleSetting.Builder()
        .name("scale")
        .description("Scale of the displayed text.")
        .defaultValue(1)
        .min(0.1)
        .sliderMax(4)
        .build()
    );

    private final Setting<Integer> maxDistance = sgGeneral.add(new IntSetting.Builder()
        .name("max-distance")
        .description("Maximum distance to render spawner info.")
        .defaultValue(64)
        .min(1)
        .sliderMax(128)
        .build()
    );

    private final Vector3d pos = new Vector3d();

    public SpawnerInfo() {
        super(Categories.Render, "spawner-info", "Displays the NBT data of spawners above them.");
    }

    @EventHandler
    private void onRender2D(Render2DEvent event) {
        TextRenderer text = TextRenderer.get();

        for (BlockEntity be : Utils.blockEntities()) {
            if (!(be instanceof MobSpawnerBlockEntity spawner)) continue;

            BlockPos blockPos = be.getPos();
            Vec3d vec3d = Vec3d.ofCenter(blockPos);
            if (PlayerUtils.distanceToCamera(vec3d.x, vec3d.y, vec3d.z) > maxDistance.get()) continue;

            NbtCompound tag = spawner.createNbt(mc.world.getRegistryManager());
            String nbt = tag.toString();

            pos.set(vec3d.x, vec3d.y + 1.0, vec3d.z);
            if (!NametagUtils.to2D(pos, scale.get())) continue;

            NametagUtils.begin(pos, event.drawContext);
            text.begin(scale.get(), true, true);
            double x = -text.getWidth(nbt, true) / 2.0;
            text.render(nbt, x, 0, Color.WHITE, true);
            text.end();
            NametagUtils.end(event.drawContext);
        }
    }
}
