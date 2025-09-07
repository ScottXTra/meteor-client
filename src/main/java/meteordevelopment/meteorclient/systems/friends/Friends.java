/*
 * This file is part of the Meteor Client distribution (https://github.com/MeteorDevelopment/meteor-client).
 * Copyright (c) Meteor Development.
 */

package meteordevelopment.meteorclient.systems.friends;

import com.mojang.util.UndashedUuid;
import meteordevelopment.meteorclient.systems.System;
import meteordevelopment.meteorclient.systems.Systems;
import meteordevelopment.meteorclient.utils.misc.NbtUtils;
import meteordevelopment.meteorclient.utils.network.MeteorExecutor;
import net.minecraft.client.network.PlayerListEntry;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtElement;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

public class Friends extends System<Friends> implements Iterable<Friend> {
    private final List<Friend> friends = new ArrayList<>();

    public Friends() {
        super("friends");
    }

    public static Friends get() {
        return Systems.get(Friends.class);
    }

    public boolean add(Friend friend) {
        if (friend.name.isEmpty() || friend.name.contains(" ")) return false;

        if (!friends.contains(friend)) {
            friends.add(friend);
            save();

            return true;
        }

        return false;
    }

    public boolean remove(Friend friend) {
        if (friends.remove(friend)) {
            save();
            return true;
        }

        return false;
    }

    public Friend get(String name) {
        for (Friend friend : friends) {
            if (friend.name.equalsIgnoreCase(name)) return friend;
        }
        return null;
    }

    public Friend get(UUID id) {
        for (Friend friend : friends) {
            if (friend.getId() != null && friend.getId().equals(id)) return friend;
        }
        return null;
    }

    public Friend get(PlayerEntity player) {
        Friend friend = get(player.getUuid());
        return friend != null ? friend : get(player.getName().getString());
    }

    public Friend get(PlayerListEntry player) {
        Friend friend = get(player.getProfile().getId());
        return friend != null ? friend : get(player.getProfile().getName());
    }

    public boolean isFriend(PlayerEntity player) {
        return player != null && get(player) != null;
    }

    public boolean isFriend(PlayerListEntry player) {
        return get(player) != null;
    }

    public boolean shouldAttack(PlayerEntity player) {
        return !isFriend(player);
    }

    public int count() {
        return friends.size();
    }

    public boolean isEmpty() {
        return friends.isEmpty();
    }

    @Override
    public @NotNull Iterator<Friend> iterator() {
        return friends.iterator();
    }

    @Override
    public NbtCompound toTag() {
        NbtCompound tag = new NbtCompound();

        tag.put("friends", NbtUtils.listToTag(friends));

        return tag;
    }

    @Override
    public Friends fromTag(NbtCompound tag) {
        friends.clear();

        for (NbtElement itemTag : tag.getListOrEmpty("friends")) {
            NbtCompound friendTag = (NbtCompound) itemTag;
            if (!friendTag.contains("name")) continue;

            String name = friendTag.getString("name", "");
            String uuid = friendTag.getString("id", "");
            Friend friend = !uuid.isBlank()
                ? new Friend(name, UndashedUuid.fromStringLenient(uuid))
                : new Friend(name);

            if (friend.getId() != null) {
                if (get(friend.getId()) != null) continue;
            }
            else if (get(name) != null) continue;

            friends.add(friend);
        }

        Collections.sort(friends);

        MeteorExecutor.execute(() -> friends.forEach(Friend::updateInfo));

        return this;
    }
}
