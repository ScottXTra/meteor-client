/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.settings;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import meteordevelopment.meteorclient.MeteorClient;
import net.minecraft.client.network.PlayerListEntry;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtElement;
import net.minecraft.nbt.NbtList;
import net.minecraft.nbt.NbtString;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;

public class PlayerListSetting extends Setting<Set<String>> {
    public final Predicate<String> filter;

    public PlayerListSetting(String name, String description, Set<String> defaultValue, Consumer<Set<String>> onChanged, Consumer<Setting<Set<String>>> onModuleActivated, Predicate<String> filter, IVisible visible) {
        super(name, description, defaultValue, onChanged, onModuleActivated, visible);
        this.filter = filter;
    }

    @Override
    protected void resetImpl() {
        value = new ObjectOpenHashSet<>(defaultValue);
    }

    @Override
    protected Set<String> parseImpl(String str) {
        String[] values = str.split(",");
        Set<String> names = new ObjectOpenHashSet<>(values.length);
        for (String value : values) {
            String name = value.trim();
            if (name.isEmpty()) continue;
            if (filter == null || filter.test(name)) names.add(name);
        }
        return names;
    }

    @Override
    protected boolean isValueValid(Set<String> value) {
        return true;
    }

    @Override
    public List<String> getSuggestions() {
        List<String> suggestions = new ArrayList<>();
        if (MeteorClient.mc != null && MeteorClient.mc.getNetworkHandler() != null) {
            for (PlayerListEntry entry : MeteorClient.mc.getNetworkHandler().getPlayerList()) {
                String name = entry.getProfile().getName();
                if (filter == null || filter.test(name)) suggestions.add(name);
            }
        }
        return suggestions;
    }

    @Override
    public NbtCompound save(NbtCompound tag) {
        NbtList valueTag = new NbtList();
        for (String name : get()) {
            valueTag.add(NbtString.of(name));
        }
        tag.put("value", valueTag);
        return tag;
    }

    @Override
    public Set<String> load(NbtCompound tag) {
        get().clear();
        NbtElement valueTag = tag.get("value");
        if (valueTag instanceof NbtList list) {
            for (NbtElement t : list) {
                String name = t.asString().orElse("");
                if (filter == null || filter.test(name)) get().add(name);
            }
        }
        return get();
    }

    public static class Builder extends SettingBuilder<Builder, Set<String>, PlayerListSetting> {
        private Predicate<String> filter;

        public Builder() {
            super(new ObjectOpenHashSet<>(0));
        }

        public Builder defaultValue(String... defaults) {
            ObjectOpenHashSet<String> set = new ObjectOpenHashSet<>(defaults == null ? new String[0] : defaults);
            return defaultValue(set);
        }

        public Builder filter(Predicate<String> filter) {
            this.filter = filter;
            return this;
        }

        @Override
        public PlayerListSetting build() {
            return new PlayerListSetting(name, description, defaultValue, onChanged, onModuleActivated, filter, visible);
        }
    }
}
