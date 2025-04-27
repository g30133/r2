package com.example.steve1;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.io.File;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.time.Instant;
import java.time.Duration;
import java.awt.event.KeyEvent;
import java.nio.charset.StandardCharsets;

import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHandler;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.gui.screens.inventory.CraftingScreen;
import net.minecraft.client.gui.screens.DeathScreen;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.inventory.ClickType;
import net.minecraft.world.inventory.Slot;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.Container;
import net.minecraft.core.Registry;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.resources.ResourceLocation;

import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.entity.player.PlayerContainerEvent;
import net.minecraftforge.client.event.ContainerScreenEvent;
import net.minecraftforge.client.event.RenderLevelStageEvent;
import net.minecraftforge.client.event.RenderGuiOverlayEvent;
import net.minecraftforge.client.event.RenderGuiEvent;
import net.minecraftforge.client.event.ScreenEvent;
import net.minecraftforge.client.event.InputEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.eventbus.api.EventPriority;
import net.minecraftforge.fml.common.Mod;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.platform.InputConstants;

import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL12;
import org.lwjgl.opengl.GL30;
import org.lwjgl.BufferUtils;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWCursorPosCallbackI;

@Mod.EventBusSubscriber
@Mod(Steve1.MOD_ID)
public class Steve1 {
    public static final String MOD_ID = "steve1";


    private static final int WIDTH = 640, HEIGHT = 360;

	private static double cursorX = WIDTH / 2;
	private static double cursorY = HEIGHT / 2;

    private static final int FRAMETYPE_PLAY = 0;
    private static final int FRAMETYPE_INVENTORY = 1;
    private static final int FRAMETYPE_CRAFTING = 2;
    private static final int FRAMETYPE_DEATH = 3;

    
    private static final int HEADER_SIZE = 4 + 4 + 4;
    private static final int FRAME_SIZE = 4 + 4 + WIDTH * HEIGHT * 4;
    private static final int CONTROL_SIZE = 4 + 4 + 4;
    private static final int TIMING_SIZE = 8 + 8 + 8 + 8 + 8 + 8;

    private static final int INVENTORY_SIZE = 200 * 20;
    
    private static final int SHMBUF_FRAME_SIZE = HEADER_SIZE + FRAME_SIZE;
    private static final int SHMBUF_CONTROL_SIZE = HEADER_SIZE + CONTROL_SIZE;
    private static final int SHMBUF_TIMING_SIZE = HEADER_SIZE + TIMING_SIZE;
    private static final int SHMBUF_INVENTORY_SIZE = HEADER_SIZE + INVENTORY_SIZE;
    
    private static final long TARGET_FRAMEINTERVAL_MILLISEC = 50;
    private static final String SHMBUF_FRAME_PATH = "/dev/shm/minecraft_frame";
    private static final String SHMBUF_CONTROL_PATH = "/dev/shm/minecraft_control";
    private static final String SHMBUF_TIMING_PATH = "/dev/shm/minecraft_timing";
    private static final String SHMBUF_INVENTORY_PATH = "/dev/shm/minecraft_inventory";
    private static final ResourceLocation CUSTOM_CURSOR = new ResourceLocation(MOD_ID, "textures/gui/left_ptr_000.png");
    private static final ResourceLocation CUSTOM_CURSOR2 = new ResourceLocation(MOD_ID, "textures/gui/arrow_000.png");
    private static ByteBuffer bytebuf;
    private static MappedByteBuffer shmbuf_frame;
    private static MappedByteBuffer shmbuf_control;
    private static MappedByteBuffer shmbuf_timing;
    private static MappedByteBuffer shmbuf_inventory;
    private static int framecounter;
    private static int frametype;
    private static int timingcounter;
    private static int inventorycounter;
    private static long tick_logic = 0;
    private static long tick_render = 0;
    private static long tick_playscreen_1 = 0;
    private static long tick_playscreen_2 = 0;
    private static long tick_menuscreen_1 = 0;
    private static long tick_menuscreen_2 = 0;
    private static int prev_inventory_control = 0;
    private static Minecraft mc = Minecraft.getInstance();

    private static final int[] binIx2Pixels = {
	-66,
	-38,
	-21,
	-10,
	-4,
	0,
	4,
	10,
	21,
	38,
	66
    };
    
    
    private static final double[] binIx2Angle = {
        -10.0,
         -5.809483127522302,
         -3.2153691330919005,
         -1.6094986352788734,
         -0.6153942662021781,
          0.0,
          0.6153942662021781,
          1.6094986352788734,
          3.2153691330919005,
          5.809483127522302,
         10.0
    };
    
    private static final KeyMapping[] buttonBitsIx2KeyMapping = {
	mc.options.keyAttack,
	mc.options.keyDown,
	mc.options.keyUp,
	mc.options.keyJump,
	mc.options.keyLeft,
	mc.options.keyRight,
	mc.options.keyShift,
	mc.options.keySprint,
	mc.options.keyUse,
	mc.options.keyDrop,
	mc.options.keyInventory,
	mc.options.keyHotbarSlots[0],
	mc.options.keyHotbarSlots[1],
	mc.options.keyHotbarSlots[2],
	mc.options.keyHotbarSlots[3],
	mc.options.keyHotbarSlots[4],
	mc.options.keyHotbarSlots[5],
	mc.options.keyHotbarSlots[6],
	mc.options.keyHotbarSlots[7],
	mc.options.keyHotbarSlots[8]
    };

    public Steve1() {
        try {
	    File f = null;
	    RandomAccessFile raf = null;
	    
	    f = new File(SHMBUF_FRAME_PATH);
	    if (f.exists() == false) {
		raf = new RandomAccessFile(f, "rw");
		raf.setLength(SHMBUF_FRAME_SIZE);
		raf.close();
	    }
	    
      	    f = new File(SHMBUF_CONTROL_PATH);
	    if (f.exists() == false) {
		raf = new RandomAccessFile(f, "rw");
		raf.setLength(SHMBUF_CONTROL_SIZE);
		raf.close();
	    }
	    
      	    f = new File(SHMBUF_TIMING_PATH);
	    if (f.exists() == false) {
		raf = new RandomAccessFile(f, "rw");
		raf.setLength(SHMBUF_TIMING_SIZE);
		raf.close();
	    }

	    f = new File(SHMBUF_INVENTORY_PATH);
	    if (f.exists() == false) {

		raf = new RandomAccessFile(f, "rw");
		raf.setLength(SHMBUF_INVENTORY_SIZE);
		raf.close();
	    }
	    
	    raf = new RandomAccessFile(SHMBUF_FRAME_PATH, "rw");
        shmbuf_frame = raf.getChannel().map(FileChannel.MapMode.READ_WRITE, 0, SHMBUF_FRAME_SIZE);
	    
	    raf = new RandomAccessFile(SHMBUF_CONTROL_PATH, "rw");
        shmbuf_control = raf.getChannel().map(FileChannel.MapMode.READ_WRITE, 0, SHMBUF_CONTROL_SIZE);
	    
	    raf = new RandomAccessFile(SHMBUF_TIMING_PATH, "rw");
        shmbuf_timing = raf.getChannel().map(FileChannel.MapMode.READ_WRITE, 0, SHMBUF_TIMING_SIZE);

	    raf = new RandomAccessFile(SHMBUF_INVENTORY_PATH, "rw");
	    shmbuf_inventory = raf.getChannel().map(FileChannel.MapMode.READ_WRITE, 0, SHMBUF_INVENTORY_SIZE);
	    
        bytebuf = BufferUtils.createByteBuffer(FRAME_SIZE - 8);
	    framecounter = 0;
	    inventorycounter = 0;

        } catch (Exception e) {
            e.printStackTrace();
        }
	//listMouseHandlerFields();
        System.out.println("Steve1 Loaded!");
    }

    @SubscribeEvent
    public static void onClientTick(TickEvent.ClientTickEvent event) {
	//System.out.println(framecounter + "|" + frametype + " onClientTick 1");
	if (event.phase != TickEvent.Phase.START) return;

	//System.out.println(framecounter + "|" + frametype + " onClientTick 2");
	tick_logic = System.currentTimeMillis();

	timingcounter = timingcounter + 1;
	sendTiming();
	
       	//consumeTime(TARGET_FRAMEINTERVAL_MILLISEC - Math.min(t2 - t0, t3 - t0));
    }

    @SubscribeEvent
    public static void onRender(RenderLevelStageEvent event) {
	//System.out.println(framecounter + "|" + frametype + " onRender 1");
	if (event.getStage().toString().contains("after_weather") == false) return;

	//System.out.println(framecounter + "|" + frametype + " onRender 2");
	tick_render = System.currentTimeMillis();
    }

    @SubscribeEvent
    public static void onRenderGuiOverlay(RenderGuiOverlayEvent.Post event) {
	//System.out.println(framecounter + "|" + frametype + " onRenderGuiOverlay 1");
	if (mc.screen != null) return;

	//System.out.println(framecounter + "|" + frametype + " onRenderGuiOverlay 2");
	if (event.getOverlay().toString().contains("player_list") == false) return;

	//System.out.println(framecounter + "|" + frametype + " onRenderGuiOverlay 3");

	long tick = System.currentTimeMillis();
	long delta = tick - tick_playscreen_1;
	tick_playscreen_1 = tick;

	framecounter = framecounter + 1;
	frametype = FRAMETYPE_PLAY;

	inventorycounter = inventorycounter + 1;
	sendInventory();
	
        sendFrame();
        recvControl();

	tick_playscreen_2 = System.currentTimeMillis();
	
	consumeTime(TARGET_FRAMEINTERVAL_MILLISEC - Math.min(delta, tick_playscreen_1 - tick_menuscreen_1));
    }

    @SubscribeEvent(priority = EventPriority.LOWEST)
    public static void onRenderGui(RenderGuiEvent.Post event) {
	//System.out.println(framecounter + "|" + frametype + " onRenderGui 1");

	if (mc.screen == null) return;

	//System.out.println(framecounter + "|" + frametype + " onRenderGui 2");	
	if (mc.screen instanceof InventoryScreen) {
	    frametype = FRAMETYPE_INVENTORY;
	}
	else if (mc.screen instanceof CraftingScreen)  {
	    frametype = FRAMETYPE_CRAFTING;
	}
	else if (mc.screen instanceof DeathScreen) {
	    frametype = FRAMETYPE_DEATH;
	} else {
	    return;
	}

	//System.out.println(framecounter + "|" + frametype + " onRenderGui 3");

	// render mouse cursor
	/*
	RenderSystem.setShaderTexture(0, CUSTOM_CURSOR2);
	mc.screen.blit(event.getPoseStack(),
		       (int)mc.mouseHandler.xpos(),
		       (int)mc.mouseHandler.ypos(),
		       6, 4, 10, 16, 32, 32);
	*/
    }
    
    @SubscribeEvent
    public static void onScreenRender(ScreenEvent.Render.Post event) {
	//System.out.println(framecounter + "|" + frametype + " onScreenEvent.Render 1");	

	if (mc.screen == null) return;

	//System.out.println(framecounter + "|" + frametype + " onScreenEvent.Render 2");
	if (mc.screen instanceof InventoryScreen) {
	    frametype = FRAMETYPE_INVENTORY;
	}
	else if (mc.screen instanceof CraftingScreen)  {
	    frametype = FRAMETYPE_CRAFTING;
	}
	else if (mc.screen instanceof DeathScreen) {
	    frametype = FRAMETYPE_DEATH;
	} else {
	    return;
	}
	
	//System.out.println(framecounter + "|" + frametype + " onScreenEvent.Render 3");

	long tick = System.currentTimeMillis();
	long delta = tick - tick_menuscreen_1;
	tick_menuscreen_1 = tick;

	// render mouse cursor
	/*
	RenderSystem.setShaderTexture(0, CUSTOM_CURSOR2);
	mc.screen.blit(event.getPoseStack(),
		       (int)mc.mouseHandler.xpos(),
		       (int)mc.mouseHandler.ypos(),
		       6, 4, 10, 16, 32, 32);
	*/

	framecounter = framecounter + 1;
	
	inventorycounter = inventorycounter + 1;
	sendInventory();

        sendFrame();
	recvControl();

	tick_menuscreen_2 = System.currentTimeMillis();

	consumeTime(TARGET_FRAMEINTERVAL_MILLISEC - Math.min(delta, tick_menuscreen_1 - tick_playscreen_1));
    }

    @SubscribeEvent
    public static void onScreenOpen(ScreenEvent.Opening event) {
		if (event.getScreen() instanceof InventoryScreen ||
			event.getScreen() instanceof CraftingScreen ||
			event.getScreen() instanceof DeathScreen) {
			//System.out.println("InventoryScreenOpening");
			// Reset to GUI center
			cursorX = WIDTH / 2;
			cursorY = HEIGHT / 2;
			setCursorPos(WIDTH/2, HEIGHT/2);
		}
    }

    @SubscribeEvent
    public static void onScreenClose(ScreenEvent.Closing event) {
	if (event.getScreen() instanceof InventoryScreen) {
	    //System.out.println("InventoryScreenClosing");
	}
    }

    private static void consumeTime(long delta) {
	if (delta > 0) {
	    try {
		Thread.sleep(delta); // in millisec
		//System.out.println("Thread.sleep:"+ delta + " millisec");
	    } catch (InterruptedException e) {
		e.printStackTrace();
	    }
	}
    }

    private static void dumpFrame(String tag, int framecounter, int frametype, int[] buttonBits, int camera0, int camera1) {
	String header = tag + "[" + framecounter + "," + frametype + "]";
	if (tag == "sent") {
	    System.out.println(header);
	} else {
	    String buttons = "[";
	    for (int ix = 0; ix < buttonBits.length; ix++) {
		if (ix == 0) {
		    buttons = buttons + buttonBits[ix];
		} else if (ix == 10) {
		    buttons = buttons + "|" + buttonBits[ix];
		} else {
		    buttons = buttons + buttonBits[ix];
		}
	    }
	    buttons = buttons + "]";
	    String camera = "[" + camera0 + "," + camera1 + "]";
	    System.out.println(header + buttons + camera);
	}
    }
    
    public static String getInventory() {
		Map<String, Integer> itemCounts = new HashMap<>();
		
		// Access the player's inventory
		for (int i = 0; i < mc.player.getInventory().getContainerSize(); i++) {
			ItemStack stack = mc.player.getInventory().getItem(i);
			if (!stack.isEmpty()) {
			String name = stack.getDisplayName().getString();
			int count = stack.getCount();
			itemCounts.put(name, itemCounts.getOrDefault(name, 0) + count);
			}
		}

		StringBuilder sb = new StringBuilder();
		for (Map.Entry<String, Integer> entry : itemCounts.entrySet()) {
			sb.append(entry.getKey())
			.append(" x")
			.append(entry.getValue())
			.append("\n");
		}
		return sb.toString().trim();
    }

    private static void sendTiming() {
		shmbuf_timing.position(HEADER_SIZE);
		shmbuf_timing.putLong(HEADER_SIZE, tick_logic);
		shmbuf_timing.putLong(HEADER_SIZE + 8, tick_render);
		shmbuf_timing.putLong(HEADER_SIZE + 16, tick_playscreen_1);
		shmbuf_timing.putLong(HEADER_SIZE + 24, tick_playscreen_2);
		shmbuf_timing.putLong(HEADER_SIZE + 32, tick_menuscreen_1);
		shmbuf_timing.putLong(HEADER_SIZE + 40, tick_menuscreen_2);
		shmbuf_timing.putInt(8, frametype);
		shmbuf_timing.putInt(4, framecounter);
		shmbuf_timing.force();
		shmbuf_timing.putInt(0, timingcounter);
		shmbuf_timing.force();
    }

    private static void sendInventory() {	
		shmbuf_inventory.position(HEADER_SIZE);

		// clear buffer
		for (int i = 0; i < INVENTORY_SIZE; i++) {
			shmbuf_inventory.put((byte) 0);
		}

		shmbuf_inventory.position(HEADER_SIZE);
		String inventoryString = getInventory();
		byte[] strBytes = inventoryString.getBytes(StandardCharsets.UTF_8);
		shmbuf_inventory.put(strBytes);

		shmbuf_inventory.putInt(8, frametype);
		shmbuf_inventory.putInt(4, framecounter);
		shmbuf_inventory.force();
		shmbuf_inventory.putInt(0, inventorycounter);
		shmbuf_inventory.force();
    }

    private static void sendFrame() {
        //System.out.println("sendFrame():" + framecounter);

		bytebuf.clear();

        GL11.glReadPixels(0, 0, WIDTH, HEIGHT, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, bytebuf);

		// framecounter 4 | frametype 4 | framedata 640x360x4
        shmbuf_frame.position(HEADER_SIZE + 4 + 4);
        shmbuf_frame.put(bytebuf);

		shmbuf_frame.putInt(HEADER_SIZE + 4, (int)mc.mouseHandler.ypos());
		shmbuf_frame.putInt(HEADER_SIZE, (int)mc.mouseHandler.xpos());
		
		shmbuf_frame.putInt(4, frametype);
		shmbuf_frame.force();

		shmbuf_frame.putInt(0, framecounter);
		shmbuf_frame.force();

		//dumpFrame("sent", framecounter, frametype, new int[0], -1, -1);
		//dumpInventory();
    }

    private static void recvControl() {
		//System.out.println("recvControl():" + framecounter);
		while (true) {
			try {
				Thread.sleep(1); // 1ms
			} catch (InterruptedException e) {
				e.printStackTrace();
	    	}
	    
			shmbuf_control.position(0);
			int header = shmbuf_control.getInt();
			if (header != framecounter)
			continue;

			shmbuf_control.position(4);
			frametype = shmbuf_control.getInt();

			shmbuf_control.position(HEADER_SIZE);
			int buttonControl = shmbuf_control.getInt();

			shmbuf_control.position(HEADER_SIZE + 4);
			int cameraControl = shmbuf_control.getInt();
			
			shmbuf_control.position(HEADER_SIZE + 8);
			float cameraFactor = shmbuf_control.getFloat();
			
            int[] buttonBits = int2bits(buttonControl, 20);
            int camera_pitch_ix = cameraControl / 11;
	    	int camera_yaw_ix = cameraControl % 11;

			//dumpFrame("recv", framecounter, frametype, buttonBits, camera_pitch, camera_yaw);

			injectCameraControl(camera_pitch_ix, camera_yaw_ix, cameraFactor);
			injectButtonControl(buttonBits);
		
			break;
		}
    }

    public static int[] int2bits(int n, int bitLength) {
        int[] binaryArray = new int[bitLength];
        for (int i = bitLength - 1; i >= 0; i--) {
            binaryArray[i] = (n >> (bitLength - 1 - i)) & 1; // Extract each bit
        }
	return binaryArray;
    }

    public static void injectButtonControl(int[] buttonBits) {
        for (int ix = 0; ix < buttonBits.length; ix++) {
	    int control = buttonBits[ix];
	    injectButtonControlKey(ix, control);
	}
    }

    private static void injectButtonControlKey(int ix, int control) {
	//System.out.println("injectButtonControlKey" + " ix:" + ix + " control:" + control);
        KeyMapping keymapping = buttonBitsIx2KeyMapping[ix];

	switch (ix) {
	case 0: // attack, left mouse button
	case 8: // use, right mouse button
	    if (frametype == FRAMETYPE_PLAY) {
		if (control == 1) {
		    keymapping.setDown(true);
		    keymapping.click(keymapping.getKey());
		}
		else {
		    keymapping.setDown(false);
		}
	    }
	    else { // FRAMETYPE_GAMEMENU
		if (control == 1) {
		    double xpos = mc.mouseHandler.xpos();
		    double ypos = mc.mouseHandler.ypos();
		    ItemStack item = mc.player.containerMenu.getCarried();
		    if (item.isEmpty()) {
			mc.screen.mouseClicked(xpos, ypos, ix == 0 ? 0 : 1);
		    }
		    else {
			mc.screen.mouseReleased(xpos, ypos, ix == 0 ? 0 : 1);
		    }
		}
	    }
	    break;
	case 1: // back, s
	case 2: // forward, w
	case 3: // jump, space
	case 4: // left, a
	case 5: // right, d
	    if (frametype == FRAMETYPE_PLAY) {
		if (control == 1) {
		    keymapping.setDown(true);
		    keymapping.click(keymapping.getKey());
		}
		else {
		    keymapping.setDown(false);
		}
	    }
	    break;
	case 6: // sneak, shift
	case 7: // sprintk, control
	case 9: // drop, q
	    if (frametype == FRAMETYPE_PLAY) {
		if (control == 1) {
		    keymapping.setDown(true);
		    keymapping.click(keymapping.getKey());
		}
		else {
		    keymapping.setDown(false);
		}
	    }
	    else {
		if (control == 1) {
		    mc.screen.keyPressed(keymapping.getKey().getValue(), 0, 0);
		}
	    }
	    break;
	case 10: // inventory, e
	    //if (control == 1) {
	    if (control == 1 && prev_inventory_control == 0) {
		if (frametype == FRAMETYPE_PLAY) {
		    keymapping.setDown(true);
		    keymapping.click(keymapping.getKey());
		    keymapping.setDown(false);
		}
		else {
		    mc.screen.keyPressed(keymapping.getKey().getValue(), 0, 0);
		}
	    }
	    prev_inventory_control = control;
	    break;
	default: // hotbar 1 ~ 9
	    if (control == 1) {
		keymapping.setDown(true);
		keymapping.click(keymapping.getKey());
		keymapping.setDown(false);
	    }
	    break;
	}
    }

    private static void injectCameraControl(int camera_pitch_ix, int camera_yaw_ix, float cameraFactor) {
		if (frametype == FRAMETYPE_PLAY) {
			float dy = (float)binIx2Angle[camera_pitch_ix];
			float dx = (float)binIx2Angle[camera_yaw_ix];

			// option 1: doesn't work somehow
			//mc.player.turn(dx, dy);

			// option 2:
			float newXRot = mc.player.getXRot() + dy; // XRot is for vertical move
			float newYRot = mc.player.getYRot() + dx; // YRot is for horizontal move
			mc.player.setXRot(newXRot);
			mc.player.setYRot(newYRot);
			mc.player.xRotO = newXRot; // for smooth animation
			mc.player.yRotO = newYRot; // for smooth animation
		}
		else {
			// double xpos = mc.mouseHandler.xpos();
			// double ypos = mc.mouseHandler.ypos();
			// int dy = binIx2Pixels[camera_pitch_ix];
			// int dx = binIx2Pixels[camera_yaw_ix];
			
			// RenderSystem.recordRenderCall(() ->
			// 	GLFW.glfwSetCursorPos(mc.getWindow().getWindow(), xpos+dx, ypos+dy));
			
			// setCursorPos(xpos + dx, ypos + dy);
			// mc.screen.mouseMoved(xpos + dx, ypos + dy);
			int dy = binIx2Pixels[camera_pitch_ix];
			int dx = binIx2Pixels[camera_yaw_ix];

			cursorX += dx;
			cursorY += dy;

			// Clamp within screen bounds
			cursorX = Math.max(0, Math.min(cursorX, WIDTH));
			cursorY = Math.max(0, Math.min(cursorY, HEIGHT));

			RenderSystem.recordRenderCall(() ->
				GLFW.glfwSetCursorPos(mc.getWindow().getWindow(), cursorX, cursorY));

			setCursorPos(cursorX, cursorY);
			mc.screen.mouseMoved(cursorX, cursorY);
		}
    }

    private static void setCursorPos(double x, double y) {
	try {
	    double xx = Math.max(0, Math.min(x, WIDTH));
	    double yy = Math.max(0, Math.min(y, HEIGHT));
	    Field xposField = MouseHandler.class.getDeclaredField("f_91507_");
	    Field yposField = MouseHandler.class.getDeclaredField("f_91508_");
	    xposField.setAccessible(true);
	    yposField.setAccessible(true);
	    xposField.set(mc.mouseHandler, xx);
	    yposField.set(mc.mouseHandler, yy);
	} catch (Exception e) {
	    e.printStackTrace();
	}
    }
    
    private static void listMouseHandlerFields() {
	System.out.println("======== listMouseHandlerFields() ========");
	for (Field field : mc.mouseHandler.getClass().getDeclaredFields()) {
	    if (field.getType() == double.class) {
		field.setAccessible(true);
		try {
		    System.out.println("Field:" + field.getName() + " | Value:" + field.getDouble(mc.mouseHandler));
		} catch (Exception e) {
		    System.out.println("Field:" + field.getName() + " | Unable to read value");
		}
	    }
	}
	System.out.println("===========================================");
    }
}

