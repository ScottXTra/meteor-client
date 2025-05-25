/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.settings;

import meteordevelopment.meteorclient.gui.utils.CharFilter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * A StringListSetting that provides suggestions based on a supplier of player names.
 */
public class PlayerListSetting extends StringListSetting {
    private final Supplier<Collection<String>> supplier;

    public PlayerListSetting(String name, String description, List<String> defaultValue,
                             Consumer<List<String>> onChanged,
                             Consumer<Setting<List<String>>> onModuleActivated,
                             IVisible visible,
                             Supplier<Collection<String>> supplier) {
        super(name, description, defaultValue, onChanged, onModuleActivated, visible, null, null);
        this.supplier = supplier;
    }

    @Override
    public List<String> getSuggestions() {
        return new ArrayList<>(supplier.get());
    }

    public static class Builder extends SettingBuilder<Builder, List<String>, PlayerListSetting> {
        private Supplier<Collection<String>> supplier;

        public Builder() {
            super(new ArrayList<>(0));
        }

        public Builder supplier(Supplier<Collection<String>> supplier) {
            this.supplier = supplier;
            return this;
        }

        public Builder defaultValue(String... defaults) {
            return defaultValue(defaults != null ? List.of(defaults) : new ArrayList<>(0));
        }

        @Override
        public PlayerListSetting build() {
            return new PlayerListSetting(name, description, defaultValue, onChanged, onModuleActivated, visible, supplier);
        }
    }
}
