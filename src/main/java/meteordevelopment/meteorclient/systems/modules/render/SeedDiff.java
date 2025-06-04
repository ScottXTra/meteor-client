package meteordevelopment.meteorclient.systems.modules.render;

import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import meteordevelopment.meteorclient.events.render.Render3DEvent;
import meteordevelopment.meteorclient.events.world.BlockUpdateEvent;
import meteordevelopment.meteorclient.events.world.ChunkDataEvent;
import meteordevelopment.meteorclient.renderer.ShapeMode;
import meteordevelopment.meteorclient.settings.StringSetting;
import meteordevelopment.meteorclient.settings.Setting;
import meteordevelopment.meteorclient.settings.SettingGroup;
import meteordevelopment.meteorclient.settings.IntSetting;
import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import meteordevelopment.meteorclient.utils.Utils;
import meteordevelopment.meteorclient.utils.render.color.SettingColor;
import meteordevelopment.meteorclient.settings.ColorSetting;
import meteordevelopment.orbit.EventHandler;
import meteordevelopment.meteorclient.mixin.ClientPlayNetworkHandlerAccessor;
import net.minecraft.block.BlockState;
import net.minecraft.client.network.ClientPlayNetworkHandler;
import net.minecraft.registry.DynamicRegistryManager;
import net.minecraft.registry.RegistryWrapper;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.HeightLimitView;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.WorldPresets;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.dimension.DimensionOptions;

import java.util.HashSet;
import java.util.Set;

public class SeedDiff extends Module {
    private final SettingGroup sgGeneral = settings.getDefaultGroup();

    private final Setting<String> seed = sgGeneral.add(new StringSetting.Builder()
        .name("seed")
        .description("Seed used for natural terrain generation.")
        .defaultValue("0")
        .build()
    );

    private final Setting<Integer> radius = sgGeneral.add(new IntSetting.Builder()
        .name("radius")
        .description("Comparison radius around the player.")
        .defaultValue(10)
        .min(3)
        .sliderRange(3, 30)
        .build()
    );

    private final Setting<SettingColor> color = sgGeneral.add(new ColorSetting.Builder()
        .name("color")
        .description("Color of highlighted blocks.")
        .defaultValue(new SettingColor(255, 0, 0, 35))
        .build()
    );

    private final Long2ObjectMap<Set<BlockPos>> diffs = new Long2ObjectOpenHashMap<>();

    private ChunkGenerator generator;
    private NoiseConfig noiseConfig;

    public SeedDiff() {
        super(Categories.Render, "seed-diff", "Highlights blocks that differ from terrain generated with the specified seed.");
    }

    @Override
    public void onActivate() {
        diffs.clear();
        initGenerator();
        for (Chunk chunk : Utils.chunks()) compareChunk(chunk);
    }

    @EventHandler
    private void onChunkData(ChunkDataEvent event) {
        if (generator == null) initGenerator();
        compareChunk(event.chunk());
    }

    @EventHandler
    private void onBlockUpdate(BlockUpdateEvent event) {
        if (generator == null) return;
        Chunk chunk = mc.world.getChunk(event.pos.getX() >> 4, event.pos.getZ() >> 4);
        compareBlock(chunk, event.pos.toImmutable());
    }

    @EventHandler
    private void onRender(Render3DEvent event) {
        BlockPos playerPos = mc.player.getBlockPos();
        int r = radius.get();
        int rSq = r * r;

        diffs.values().forEach(set -> set.forEach(pos -> {
            int dx = pos.getX() - playerPos.getX();
            int dz = pos.getZ() - playerPos.getZ();
            if (dx * dx + dz * dz <= rSq) {
                event.renderer.box(pos, color.get(), color.get(), ShapeMode.Lines, 0);
            }
        }));
    }

    private void initGenerator() {
        ClientPlayNetworkHandler handler = mc.getNetworkHandler();
        if (handler == null) return;

        DynamicRegistryManager registry = ((ClientPlayNetworkHandlerAccessor) handler).getCombinedDynamicRegistries();
        RegistryWrapper.WrapperLookup lookup = registry;

        DimensionOptions options = WorldPresets.getDefaultOverworldOptions(lookup);
        generator = options.chunkGenerator();

        long worldSeed;
        try {
            worldSeed = Long.parseLong(seed.get());
        } catch (NumberFormatException e) {
            worldSeed = 0L;
        }

        if (generator instanceof NoiseChunkGenerator noiseGen) {
            noiseConfig = NoiseConfig.create(noiseGen.getSettings().value(), lookup.getOrThrow(net.minecraft.registry.RegistryKeys.NOISE_PARAMETERS), worldSeed);
        }
    }

    private void compareChunk(Chunk chunk) {
        ChunkPos pos = chunk.getPos();
        Set<BlockPos> set = diffs.computeIfAbsent(pos.toLong(), p -> new HashSet<>());

        int startX = pos.getStartX();
        int startZ = pos.getStartZ();
        HeightLimitView view = mc.world;

        BlockPos playerPos = mc.player.getBlockPos();
        int r = radius.get();
        int rSq = r * r;

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int worldX = startX + x;
                int worldZ = startZ + z;
                int dx = worldX - playerPos.getX();
                int dz = worldZ - playerPos.getZ();
                if (dx * dx + dz * dz <= rSq) {
                    compareColumn(set, chunk, worldX, worldZ, view);
                }
            }
        }
    }

    private void compareBlock(Chunk chunk, BlockPos pos) {
        BlockPos playerPos = mc.player.getBlockPos();
        int r = radius.get();
        int dx = pos.getX() - playerPos.getX();
        int dz = pos.getZ() - playerPos.getZ();
        if (dx * dx + dz * dz > r * r) return;

        Set<BlockPos> set = diffs.computeIfAbsent(chunk.getPos().toLong(), p -> new HashSet<>());
        compareColumn(set, chunk, pos.getX(), pos.getZ(), mc.world);
    }

    private void compareColumn(Set<BlockPos> set, Chunk chunk, int x, int z, HeightLimitView view) {
        if (!(generator instanceof NoiseChunkGenerator noiseGen)) return;
        if (noiseConfig == null) return;

        int bottom = view.getBottomY();
        int top = bottom + view.getHeight();
        var sample = noiseGen.getColumnSample(x, z, view, noiseConfig);

        BlockPos.Mutable bp = new BlockPos.Mutable();
        for (int y = bottom; y < top; y++) {
            BlockState expected = sample.getState(y - bottom);
            BlockState actual = chunk.getBlockState(bp.set(x, y, z));
            BlockPos immutable = bp.toImmutable();
            if (!actual.equals(expected)) set.add(immutable);
            else set.remove(immutable);
        }
    }
}
