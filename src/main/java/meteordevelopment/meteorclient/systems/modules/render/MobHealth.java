/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.render;

import meteordevelopment.meteorclient.events.render.Render2DEvent;
import meteordevelopment.meteorclient.renderer.text.TextRenderer;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.Utils;
import meteordevelopment.meteorclient.utils.render.NametagUtils;
import meteordevelopment.meteorclient.utils.render.color.Color;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.player.PlayerEntity;
import org.joml.Vector3d;

public class MobHealth extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> scale = sgGeneral.add(new DoubleSetting.Builder()
        .name("scale")
        .description("Scale of the health text.")
        .defaultValue(1)
        .min(0.1)
        .build()
    );

    private final Vector3d pos = new Vector3d();

    public MobHealth() {
        super(Categories.Render, "mob-health", "Displays the health of mobs above their heads.");
    }

    @EventHandler
    private void onRender2D(Render2DEvent event) {
        if (mc.world == null) return;

        TextRenderer text = TextRenderer.get();

        for (Entity entity : mc.world.getEntities()) {
            if (!(entity instanceof LivingEntity) || entity instanceof PlayerEntity) continue;

            LivingEntity living = (LivingEntity) entity;

            Utils.set(pos, entity, event.tickDelta);
            pos.add(0, living.getEyeHeight(living.getPose()) + 0.5, 0);

            if (!NametagUtils.to2D(pos, scale.get())) continue;

            String health = String.format("%.1f", living.getHealth() + living.getAbsorptionAmount());

            NametagUtils.begin(pos, event.drawContext);
            double w = text.getWidth(health) / 2.0;
            text.render(health, -w, 0, Color.WHITE);
            NametagUtils.end(event.drawContext);
        }
    }
}

