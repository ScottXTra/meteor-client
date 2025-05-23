/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.world;

import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.player.PlayerUtils;
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.item.Item;
import net.minecraft.item.Items;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

public class AutoIgnite extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Double> distance = sgGeneral.add(new DoubleSetting.Builder()
        .name("distance")
        .description("How far away the mobs can be to be ignited.")
        .defaultValue(4.5)
        .min(0)
        .sliderMax(6)
        .build()
    );

    private final Setting<Boolean> rotate = sgGeneral.add(new BoolSetting.Builder()
        .name("rotate")
        .description("Automatically faces towards the mob being ignited.")
        .defaultValue(true)
        .build()
    );

    private Hand hand;

    public AutoIgnite() {
        super(Categories.World, "auto-ignite", "Automatically places fire beneath mobs when holding flint and steel.");
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (mc.world == null || mc.player == null) return;

        Item main = mc.player.getMainHandStack().getItem();
        Item off = mc.player.getOffHandStack().getItem();
        if (main != Items.FLINT_AND_STEEL && off != Items.FLINT_AND_STEEL) return;

        hand = main == Items.FLINT_AND_STEEL ? Hand.MAIN_HAND : Hand.OFF_HAND;

        for (Entity entity : mc.world.getEntities()) {
            if (!(entity instanceof LivingEntity) || entity == mc.player) continue;
            if (!PlayerUtils.isWithin(entity, distance.get())) continue;

            Block block = mc.world.getBlockState(entity.getBlockPos()).getBlock();
            if (!block.equals(Blocks.AIR) && !block.equals(Blocks.GRASS_BLOCK) && !block.equals(Blocks.TALL_GRASS)) continue;

            if (rotate.get()) Rotations.rotate(Rotations.getYaw(entity.getBlockPos()), Rotations.getPitch(entity.getBlockPos()), -100, () -> ignite(entity));
            else ignite(entity);

            return;
        }
    }

    private void ignite(Entity entity) {
        mc.interactionManager.interactBlock(mc.player, hand, new BlockHitResult(
            entity.getPos().subtract(new Vec3d(0, 1, 0)), Direction.UP, entity.getBlockPos().down(), false));
        mc.player.swingHand(hand);
    }
}
