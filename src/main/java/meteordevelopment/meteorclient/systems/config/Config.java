/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.config;

import meteordevelopment.meteorclient.MeteorClient;
import meteordevelopment.meteorclient.renderer.Fonts;
import meteordevelopment.meteorclient.renderer.text.FontFace;
import meteordevelopment.meteorclient.settings.*;
import meteordevelopment.meteorclient.systems.System;
import meteordevelopment.meteorclient.systems.Systems;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.systems.modules.combat.*;
import meteordevelopment.meteorclient.systems.modules.player.*;
import meteordevelopment.meteorclient.systems.modules.movement.*;
import meteordevelopment.meteorclient.systems.modules.movement.elytrafly.ElytraFly;
import meteordevelopment.meteorclient.systems.modules.movement.speed.Speed;
import meteordevelopment.meteorclient.systems.modules.render.Tracers;
import meteordevelopment.meteorclient.systems.modules.world.*;
import meteordevelopment.meteorclient.systems.modules.misc.*;
import meteordevelopment.meteorclient.systems.modules.misc.swarm.Swarm;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtElement;
import net.minecraft.nbt.NbtList;
import net.minecraft.nbt.NbtString;

import java.util.ArrayList;
import java.util.List;

import static meteordevelopment.meteorclient.MeteorClient.mc;

public class Config extends System<Config> {
    public final Settings settings = new Settings();

    private final SettingGroup sgVisual = settings.createGroup("Visual");
    private final SettingGroup sgModules = settings.createGroup("Modules");
    private final SettingGroup sgChat = settings.createGroup("Chat");
    private final SettingGroup sgMisc = settings.createGroup("Misc");

    // Visual

    public final Setting<Boolean> customFont = sgVisual.add(new BoolSetting.Builder()
        .name("custom-font")
        .description("Use a custom font.")
        .defaultValue(true)
        .build()
    );

    public final Setting<FontFace> font = sgVisual.add(new FontFaceSetting.Builder()
        .name("font")
        .description("Custom font to use.")
        .visible(customFont::get)
        .onChanged(Fonts::load)
        .build()
    );

    public final Setting<Double> rainbowSpeed = sgVisual.add(new DoubleSetting.Builder()
        .name("rainbow-speed")
        .description("The global rainbow speed.")
        .defaultValue(0.5)
        .range(0, 10)
        .sliderMax(5)
        .build()
    );

    public final Setting<Boolean> titleScreenCredits = sgVisual.add(new BoolSetting.Builder()
        .name("title-screen-credits")
        .description("Show Meteor credits on title screen")
        .defaultValue(true)
        .build()
    );

    public final Setting<Boolean> titleScreenSplashes = sgVisual.add(new BoolSetting.Builder()
        .name("title-screen-splashes")
        .description("Show Meteor splash texts on title screen")
        .defaultValue(true)
        .build()
    );

    public final Setting<Boolean> customWindowTitle = sgVisual.add(new BoolSetting.Builder()
        .name("custom-window-title")
        .description("Show custom text in the window title.")
        .defaultValue(false)
        .onModuleActivated(setting -> mc.updateWindowTitle())
        .onChanged(value -> mc.updateWindowTitle())
        .build()
    );

    public final Setting<String> customWindowTitleText = sgVisual.add(new StringSetting.Builder()
        .name("window-title-text")
        .description("The text it displays in the window title.")
        .visible(customWindowTitle::get)
        .defaultValue("Minecraft {mc_version} - {meteor.name} {meteor.version}")
        .onChanged(value -> mc.updateWindowTitle())
        .build()
    );

    public final Setting<SettingColor> friendColor = sgVisual.add(new ColorSetting.Builder()
        .name("friend-color")
        .description("The color used to show friends.")
        .defaultValue(new SettingColor(0, 255, 180))
        .build()
    );

    // Modules

    public final Setting<List<Module>> hiddenModules = sgModules.add(new ModuleListSetting.Builder()
        .name("hidden-modules")
        .description("Prevent these modules from being rendered as options in the clickgui.")
        .defaultValue(
            AnchorAura.class, AntiAnvil.class, AntiBed.class, AutoAnvil.class, AutoArmor.class,
            AutoCity.class, AutoEXP.class, AutoTrap.class, AutoWeb.class, BedAura.class,
            BowAimbot.class, BowSpam.class, Burrow.class, CrystalAura.class, Hitboxes.class,
            HoleFiller.class, Quiver.class, SelfAnvil.class, SelfTrap.class, SelfWeb.class,
            Surround.class,

            AntiHunger.class, AutoEat.class, AutoClicker.class, AutoFish.class, AutoGap.class,
            AutoMend.class, AutoReplenish.class, BreakDelay.class, ChestSwap.class, EXPThrower.class,
            FakePlayer.class, FastUse.class, GhostHand.class, InstantRebreak.class, LiquidInteract.class,
            MiddleClickExtra.class, NoInteract.class, Portals.class, PotionSaver.class, Reach.class,
            Rotation.class, SpeedMine.class,

            AirJump.class, Anchor.class, AntiAFK.class, AntiVoid.class, AutoJump.class,
            AutoWalk.class, AutoWasp.class, Blink.class, BoatFly.class, ClickTP.class,
            ElytraBoost.class, ElytraFly.class, EntityControl.class, EntitySpeed.class,
            FastClimb.class, Flight.class, GUIMove.class, HighJump.class, Jesus.class,
            LongJump.class, NoFall.class, NoSlow.class, Parkour.class, ReverseStep.class,
            Slippy.class, Speed.class, Spider.class, Step.class, TridentBoost.class,
            Velocity.class,

            Tracers.class,

            AutoBreed.class, AutoBrewer.class, AutoMount.class, AutoNametag.class,
            AutoShearer.class, AutoSign.class, AutoSmelter.class, BuildHeight.class,
            Collisions.class, EChestFarmer.class, HighwayBuilder.class, LiquidFiller.class,
            MountBypass.class, NoGhostBlocks.class, PacketMine.class, SpawnProofer.class,
            VeinMiner.class,

            BetterBeacons.class, BetterChat.class, DiscordPresence.class, MessageAura.class,
            NameProtect.class, Notebot.class, Swarm.class
        )
        .build()
    );

    public final Setting<Integer> moduleSearchCount = sgModules.add(new IntSetting.Builder()
        .name("module-search-count")
        .description("Amount of modules and settings to be shown in the module search bar.")
        .defaultValue(8)
        .min(1).sliderMax(12)
        .build()
    );

    public final Setting<Boolean> moduleAliases = sgModules.add(new BoolSetting.Builder()
        .name("search-module-aliases")
        .description("Whether or not module aliases will be used in the module search bar.")
        .defaultValue(true)
        .build()
    );

    // Chat

    public final Setting<String> prefix = sgChat.add(new StringSetting.Builder()
        .name("prefix")
        .description("Prefix.")
        .defaultValue(".")
        .build()
    );

    public final Setting<Boolean> chatFeedback = sgChat.add(new BoolSetting.Builder()
        .name("chat-feedback")
        .description("Sends chat feedback when meteor performs certain actions.")
        .defaultValue(true)
        .build()
    );

    public final Setting<Boolean> deleteChatFeedback = sgChat.add(new BoolSetting.Builder()
        .name("delete-chat-feedback")
        .description("Delete previous matching chat feedback to keep chat clear.")
        .visible(chatFeedback::get)
        .defaultValue(true)
        .build()
    );

    // Misc

    public final Setting<Integer> rotationHoldTicks = sgMisc.add(new IntSetting.Builder()
        .name("rotation-hold")
        .description("Hold long to hold server side rotation when not sending any packets.")
        .defaultValue(4)
        .build()
    );

    public final Setting<Boolean> useTeamColor = sgMisc.add(new BoolSetting.Builder()
        .name("use-team-color")
        .description("Uses player's team color for rendering things like esp and tracers.")
        .defaultValue(true)
        .build()
    );

    public List<String> dontShowAgainPrompts = new ArrayList<>();

    public Config() {
        super("config");
    }

    public static Config get() {
        return Systems.get(Config.class);
    }

    @Override
    public NbtCompound toTag() {
        NbtCompound tag = new NbtCompound();

        tag.putString("version", MeteorClient.VERSION.toString());
        tag.put("settings", settings.toTag());
        tag.put("dontShowAgainPrompts", listToTag(dontShowAgainPrompts));

        return tag;
    }

    @Override
    public Config fromTag(NbtCompound tag) {
        if (tag.contains("settings")) settings.fromTag(tag.getCompoundOrEmpty("settings"));
        if (tag.contains("dontShowAgainPrompts")) dontShowAgainPrompts = listFromTag(tag, "dontShowAgainPrompts");

        return this;
    }

    private NbtList listToTag(List<String> list) {
        NbtList nbt = new NbtList();
        for (String item : list) nbt.add(NbtString.of(item));
        return nbt;
    }

    private List<String> listFromTag(NbtCompound tag, String key) {
        List<String> list = new ArrayList<>();
        for (NbtElement item : tag.getListOrEmpty(key)) list.add(item.asString().orElse(""));
        return list;
    }
}
