/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.events.packets.PacketEvent;
import meteordevelopment.meteorclient.events.world.TickEvent;
import meteordevelopment.meteorclient.mixin.PlayerMoveC2SPacketAccessor;
import meteordevelopment.meteorclient.mixininterface.IPlayerMoveC2SPacket;
import meteordevelopment.meteorclient.mixininterface.IVec3d;
import meteordevelopment.meteorclient.pathing.PathManagers;
import meteordevelopment.meteorclient.settings.BoolSetting;
import meteordevelopment.meteorclient.settings.DoubleSetting;
import meteordevelopment.meteorclient.settings.EnumSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.systems.modules.Modules;
import meteordevelopment.meteorclient.utils.entity.EntityUtils;
import meteordevelopment.meteorclient.utils.entity.DamageUtils;
import meteordevelopment.meteorclient.utils.player.FindItemResult;
import meteordevelopment.meteorclient.utils.player.InvUtils;
import meteordevelopment.meteorclient.utils.player.PlayerUtils;
import meteordevelopment.meteorclient.utils.player.Rotations;
import meteordevelopment.meteorclient.utils.player.SlotUtils;
import meteordevelopment.meteorclient.utils.world.BlockUtils;
import meteordevelopment.orbit.EventHandler;
import net.minecraft.block.Block;
import net.minecraft.block.Blocks;
import net.minecraft.item.BlockItem;
import net.minecraft.item.Item;
import net.minecraft.item.Items;
import net.minecraft.item.MaceItem;
import net.minecraft.network.packet.c2s.play.PlayerMoveC2SPacket;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.RaycastContext;

import java.util.function.Predicate;

public class NoFall extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<Mode> mode = sgGeneral.add(new EnumSetting.Builder<Mode>()
        .name("mode")
        .description("The way you are saved from fall damage.")
        .defaultValue(Mode.Packet)
        .build()
    );

    private final Setting<PlacedItem> placedItem = sgGeneral.add(new EnumSetting.Builder<PlacedItem>()
        .name("placed-item")
        .description("Which block to place.")
        .defaultValue(PlacedItem.Bucket)
        .visible(() -> mode.get() == Mode.Place)
        .build()
    );

    private final Setting<PlaceMode> airPlaceMode = sgGeneral.add(new EnumSetting.Builder<PlaceMode>()
        .name("air-place-mode")
        .description("Whether place mode places before you die or before you take damage.")
        .defaultValue(PlaceMode.BeforeDeath)
        .visible(() -> mode.get() == Mode.AirPlace)
        .build()
    );

    private final Setting<Boolean> anchor = sgGeneral.add(new BoolSetting.Builder()
        .name("anchor")
        .description("Centers the player and reduces movement when using bucket or air place mode.")
        .defaultValue(true)
        .visible(() -> mode.get() != Mode.Packet)
        .build()
    );

    private final Setting<Boolean> antiBounce = sgGeneral.add(new BoolSetting.Builder()
        .name("anti-bounce")
        .description("Disables bouncing on slime-block and bed upon landing.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Boolean> pauseOnMace = sgGeneral.add(new BoolSetting.Builder()
        .name("pause-on-mace")
        .description("Pauses NoFall when using a mace.")
        .defaultValue(true)
        .build()
    );

    private final Setting<Boolean> inventoryBucket = sgGeneral.add(new BoolSetting.Builder()
        .name("inventory-bucket")
        .description("Uses a water bucket from your inventory if fall damage is high enough.")
        .defaultValue(false)
        .visible(() -> mode.get() == Mode.Place && placedItem.get() == PlacedItem.Bucket)
        .build()
    );

    private final Setting<Double> damageThreshold = sgGeneral.add(new DoubleSetting.Builder()
        .name("damage-threshold")
        .description("Minimum predicted fall damage before using the inventory bucket.")
        .defaultValue(6)
        .min(0)
        .sliderRange(0, 40)
        .visible(inventoryBucket::get)
        .build()
    );

    private boolean placedWater;
    private BlockPos targetPos;
    private int timer;
    private boolean prePathManagerNoFall;
    private boolean invSwap;
    private int invSwapSlot;
    private int invSwapSelected;

    public NoFall() {
        super(Categories.Movement, "no-fall", "Attempts to prevent you from taking fall damage.");
    }

    @Override
    public void onActivate() {
        prePathManagerNoFall = PathManagers.get().getSettings().getNoFall().get();
        if (mode.get() == Mode.Packet) PathManagers.get().getSettings().getNoFall().set(true);

        placedWater = false;
        invSwap = false;
    }

    @Override
    public void onDeactivate() {
        PathManagers.get().getSettings().getNoFall().set(prePathManagerNoFall);
        invSwap = false;
    }

    @EventHandler
    private void onSendPacket(PacketEvent.Send event) {
        if (pauseOnMace.get() && mc.player.getMainHandStack().getItem() instanceof MaceItem) return;
        if (mc.player.getAbilities().creativeMode
            || !(event.packet instanceof PlayerMoveC2SPacket)
            || mode.get() != Mode.Packet
            || ((IPlayerMoveC2SPacket) event.packet).meteor$getTag() == 1337) return;


        if (!Modules.get().isActive(Flight.class)) {
            if (mc.player.isGliding()) return;
            if (mc.player.getVelocity().y > -0.5) return;
            ((PlayerMoveC2SPacketAccessor) event.packet).setOnGround(true);
        } else {
            ((PlayerMoveC2SPacketAccessor) event.packet).setOnGround(true);
        }
    }

    @EventHandler
    private void onTick(TickEvent.Pre event) {
        if (timer > 20) {
            placedWater = false;
            timer = 0;
        }

        if (mc.player.getAbilities().creativeMode) return;
        if (pauseOnMace.get() && mc.player.getMainHandStack().getItem() instanceof MaceItem) return;

        // Airplace mode
        if (mode.get() == Mode.AirPlace) {
            // Test if fall damage setting is valid
            if (!airPlaceMode.get().test((float) mc.player.fallDistance)) return;

            // Center and place block
            if (anchor.get()) PlayerUtils.centerPlayer();

            Rotations.rotate(mc.player.getYaw(), 90, Integer.MAX_VALUE, () -> {
                double preY = mc.player.getVelocity().y;
                ((IVec3d) mc.player.getVelocity()).meteor$setY(0);

                BlockUtils.place(mc.player.getBlockPos().down(), InvUtils.findInHotbar(itemStack -> itemStack.getItem() instanceof BlockItem), false, 0, true);

                ((IVec3d) mc.player.getVelocity()).meteor$setY(preY);
            });
        }

        // Bucket mode
        else if (mode.get() == Mode.Place) {
            PlacedItem placedItem1 = mc.world.getDimension().ultrawarm() && placedItem.get() == PlacedItem.Bucket ? PlacedItem.PowderSnow : placedItem.get();
            if (mc.player.fallDistance > 3 && !EntityUtils.isAboveWater(mc.player)) {
                Item item = placedItem1.item;

                FindItemResult findItemResult = inventoryBucket.get() && placedItem1 == PlacedItem.Bucket ? InvUtils.find(item) : InvUtils.findInHotbar(item);
                if (!findItemResult.found()) return;

                if (inventoryBucket.get() && placedItem1 == PlacedItem.Bucket && DamageUtils.fallDamage(mc.player) < damageThreshold.get()) return;

                if (anchor.get()) PlayerUtils.centerPlayer();

                BlockHitResult result = mc.world.raycast(new RaycastContext(mc.player.getPos(), mc.player.getPos().subtract(0, 5, 0), RaycastContext.ShapeType.OUTLINE, RaycastContext.FluidHandling.NONE, mc.player));

                if (result != null && result.getType() == HitResult.Type.BLOCK) {
                    targetPos = result.getBlockPos().up();
                    if (placedItem1 == PlacedItem.Bucket)
                        useItemSmart(findItemResult, true, targetPos, true);
                    else {
                        useItem(findItemResult, placedItem1 == PlacedItem.PowderSnow, targetPos, false);
                    }
                }
            }

            // Remove placed
            if (placedWater) {
                timer++;
                if (mc.player.getBlockStateAtPos().getBlock() == placedItem1.block) {
                    if (inventoryBucket.get() && placedItem1 == PlacedItem.Bucket) pickUpSmart(targetPos);
                    else useItem(InvUtils.findInHotbar(Items.BUCKET), false, targetPos, true);
                } else if (mc.world.getBlockState(mc.player.getBlockPos().down()).getBlock() == Blocks.POWDER_SNOW && mc.player.fallDistance==0 && placedItem1.block==Blocks.POWDER_SNOW){
                    useItem(InvUtils.findInHotbar(Items.BUCKET), false, targetPos.down(), true);
                }
            }
        }
    }

    public boolean cancelBounce() {
        return isActive() && antiBounce.get();
    }

    private void useItem(FindItemResult item, boolean placedWater, BlockPos blockPos, boolean interactItem) {
        if (!item.found()) return;

        if (interactItem) {
            Rotations.rotate(Rotations.getYaw(blockPos), Rotations.getPitch(blockPos), 10, true, () -> {
                if (item.isOffhand()) {
                    mc.interactionManager.interactItem(mc.player, Hand.OFF_HAND);
                } else {
                    InvUtils.swap(item.slot(), true);
                    mc.interactionManager.interactItem(mc.player, Hand.MAIN_HAND);
                    InvUtils.swapBack();
                }
            });
        } else {
            BlockUtils.place(blockPos, item, true, 10, true);
        }

        this.placedWater = placedWater;
    }

    private void useItemSmart(FindItemResult item, boolean placedWater, BlockPos blockPos, boolean interactItem) {
        if (!item.found()) return;

        if (interactItem) {
            Rotations.rotate(Rotations.getYaw(blockPos), Rotations.getPitch(blockPos), 10, true, () -> {
                if (item.isOffhand()) {
                    mc.interactionManager.interactItem(mc.player, Hand.OFF_HAND);
                } else if (item.isHotbar()) {
                    InvUtils.swap(item.slot(), true);
                    mc.interactionManager.interactItem(mc.player, Hand.MAIN_HAND);
                    InvUtils.swapBack();
                } else {
                    invSwapSelected = mc.player.getInventory().getSelectedSlot();
                    invSwapSlot = item.slot();
                    InvUtils.quickSwap().fromId(SlotUtils.indexToId(invSwapSelected)).to(invSwapSlot);
                    mc.interactionManager.interactItem(mc.player, Hand.MAIN_HAND);
                    invSwap = true;
                }
            });
        } else {
            BlockUtils.place(blockPos, item, true, 10, true);
        }

        this.placedWater = placedWater;
    }

    private void pickUpSmart(BlockPos blockPos) {
        Rotations.rotate(Rotations.getYaw(blockPos), Rotations.getPitch(blockPos), 10, true, () -> {
            mc.interactionManager.interactItem(mc.player, Hand.MAIN_HAND);
            if (invSwap) {
                InvUtils.quickSwap().fromId(SlotUtils.indexToId(invSwapSelected)).to(invSwapSlot);
                invSwap = false;
            } else {
                InvUtils.swapBack();
            }
        });

        placedWater = false;
    }

    @Override
    public String getInfoString() {
        return mode.get().toString();
    }

    public enum Mode {
        Packet,
        AirPlace,
        Place
    }

    public enum PlacedItem {
        Bucket(Items.WATER_BUCKET, Blocks.WATER),
        PowderSnow(Items.POWDER_SNOW_BUCKET, Blocks.POWDER_SNOW),
        HayBale(Items.HAY_BLOCK, Blocks.HAY_BLOCK),
        Cobweb(Items.COBWEB, Blocks.COBWEB),
        SlimeBlock(Items.SLIME_BLOCK, Blocks.SLIME_BLOCK);

        private final Item item;
        private final Block block;

        PlacedItem(Item item, Block block) {
            this.item = item;
            this.block = block;
        }
    }

    public enum PlaceMode {
        BeforeDamage(height -> height > 2),
        BeforeDeath(height -> height > Math.max(PlayerUtils.getTotalHealth(), 2));

        private final Predicate<Float> fallHeight;

        PlaceMode(Predicate<Float> fallHeight) {
            this.fallHeight = fallHeight;
        }

        public boolean test(float fallheight) {
            return fallHeight.test(fallheight);
        }
    }
}
