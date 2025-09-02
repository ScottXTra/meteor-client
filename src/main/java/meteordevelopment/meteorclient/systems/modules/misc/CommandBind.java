/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.modules.misc;

import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.settings.StringSetting;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.player.ChatUtils;

public class CommandBind extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<String> command = sgGeneral.add(new StringSetting.Builder()
        .name("command")
        .description("Command to run when this module is toggled.")
        .defaultValue(".vclip 5")
        .build()
    );

    public CommandBind() {
        super(Categories.Misc, "command-bind", "Binds a custom command to this module's keybind.");
    }

    @Override
    public void onActivate() {
        if (!command.get().isBlank()) ChatUtils.sendPlayerMsg(command.get());
        toggle();
    }
}

