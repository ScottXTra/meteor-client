package meteordevelopment.meteorclient.systems.modules.movement;

import meteordevelopment.meteorclient.systems.modules.Categories;
import meteordevelopment.meteorclient.systems.modules.Module;
import net.minecraft.util.math.Vec3d;
import net.minecraft.util.math.MathHelper;

import java.io.*;

public class PythonQLearning extends Module {
    private Process process;
    private BufferedWriter writer;
    private BufferedReader reader;
    private BufferedReader errReader;
    private Thread thread;
    private Thread debugThread;
    private Vec3d goal;
    private boolean goalChanged;

    public PythonQLearning() {
        super(Categories.Movement, "python-qlearning", "Moves the player using a Python Q-learning agent.");
    }

    @Override
    public void onActivate() {
        try {
            InputStream script = PythonQLearning.class.getResourceAsStream("/scripts/qlearning_agent.py");
            if (script == null) {
                error("Could not find Python script.");
                toggle();
                return;
            }
            File temp = File.createTempFile("qlearning_agent", ".py");
            try (FileOutputStream out = new FileOutputStream(temp)) {
                script.transferTo(out);
            }
            temp.deleteOnExit();
            process = new ProcessBuilder("python", temp.getAbsolutePath()).start();
            writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            errReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        } catch (IOException e) {
            error("Failed to start Python: {}", e.getMessage());
            toggle();
            return;
        }

        thread = new Thread(this::loop, "python-qlearning-loop");
        thread.start();

        debugThread = new Thread(this::readDebug, "python-qlearning-debug");
        debugThread.start();
    }

    private void loop() {
        try {
            while (isActive() && mc.player != null && !Thread.interrupted()) {
                if (goal == null) setGoal();

                Vec3d p = mc.player.getPos();
                String msg = String.format("{\"player\":{\"x\":%f,\"y\":%f,\"z\":%f,\"yaw\":%f,\"pitch\":%f},\"goal\":{\"x\":%f,\"y\":%f,\"z\":%f}%s}\n",
                    p.x, p.y, p.z, mc.player.getYaw(), mc.player.getPitch(), goal.x, goal.y, goal.z, goalChanged ? ",\"reset\":true" : "");
                goalChanged = false;
                writer.write(msg);
                writer.flush();

                String action = reader.readLine();
                if (action == null) break;
                mc.execute(() -> applyAction(action.trim()));
                Thread.sleep(50);

                Vec3d p2 = mc.player.getPos();
                if (p2.squaredDistanceTo(goal) < 1) setGoal();
            }
        } catch (Exception e) {
            error("Python communication error: {}", e.getMessage());
        }
        mc.execute(this::toggle);
    }

    private void applyAction(String action) {
        mc.options.forwardKey.setPressed(action.equals("forward") || action.equals("sprint-forward"));
        mc.options.backKey.setPressed(action.equals("back"));
        mc.options.leftKey.setPressed(action.equals("left"));
        mc.options.rightKey.setPressed(action.equals("right"));
        mc.options.sprintKey.setPressed(action.equals("sprint-forward"));

        float yaw = mc.player.getYaw();
        float pitch = mc.player.getPitch();

        if (action.equals("look-left")) mc.player.setYaw(yaw - 5);
        else if (action.equals("look-right")) mc.player.setYaw(yaw + 5);
        else if (action.equals("look-up")) mc.player.setPitch(MathHelper.clamp(pitch - 5, -90, 90));
        else if (action.equals("look-down")) mc.player.setPitch(MathHelper.clamp(pitch + 5, -90, 90));
    }

    @Override
    public void onDeactivate() {
        mc.options.forwardKey.setPressed(false);
        mc.options.backKey.setPressed(false);
        mc.options.leftKey.setPressed(false);
        mc.options.rightKey.setPressed(false);
        mc.options.sprintKey.setPressed(false);

        if (thread != null) thread.interrupt();
        if (debugThread != null) debugThread.interrupt();
        if (process != null) process.destroy();
    }

    private void setGoal() {
        if (mc.player == null) return;
        Vec3d p = mc.player.getPos();
        double angle = Math.random() * Math.PI * 2;
        goal = new Vec3d(p.x + 12 * Math.cos(angle), p.y, p.z + 12 * Math.sin(angle));
        goalChanged = true;
    }

    private void readDebug() {
        try {
            String line;
            while ((line = errReader.readLine()) != null && !Thread.interrupted()) {
                String msg = line;
                mc.execute(() -> info(msg));
            }
        } catch (IOException ignored) {
        }
    }
}
