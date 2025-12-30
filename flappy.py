import pygame
import sys
import random
import math
import os


class Config:
    """Game configuration and constants."""
    
    # Display settings
    BASE_WIDTH = 288
    BASE_HEIGHT = 512
    SCALE = 1.5
    
    # Computed screen dimensions
    SCREEN_WIDTH = int(BASE_WIDTH * SCALE)
    SCREEN_HEIGHT = int(BASE_HEIGHT * SCALE)
    
    # Physics
    GRAVITY = 0.4 * SCALE
    FLAP_VELOCITY = -7 * SCALE
    PIPE_SPEED = 2 * SCALE
    
    # Gameplay
    GAP_SIZE = int(100 * SCALE)
    BIRD_X = int(60 * SCALE)  # Fixed X position for all birds
    
    # Genetic Algorithm
    POPULATION_SIZE = 30
    ELITE_FRACTION = 0.3
    MUTATION_RATE = 0.1
    MUTATION_STD = 0.5
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SPRITES_DIR = os.path.join(SCRIPT_DIR, "sprites")
    AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")


class Assets:
    """Handles loading and storing all game assets (sprites and sounds)."""
    
    def __init__(self):
        self.sprites = {}
        self.sounds = {}
        self._load_all()
    
    def _load_sprite(self, name: str) -> pygame.Surface:
        """Load and scale a sprite image."""
        path = os.path.join(Config.SPRITES_DIR, name)
        image = pygame.image.load(path).convert_alpha()
        new_size = (
            int(image.get_width() * Config.SCALE),
            int(image.get_height() * Config.SCALE)
        )
        return pygame.transform.scale(image, new_size)
    
    def _load_sound(self, name: str) -> pygame.mixer.Sound | None:
        """Load a sound file, returns None if loading fails."""
        path = os.path.join(Config.AUDIO_DIR, name)
        try:
            return pygame.mixer.Sound(path)
        except Exception:
            return None
    
    def _load_all(self):
        """Load all game assets."""
        # Backgrounds
        self.sprites["bg_day"] = self._load_sprite("background-day.png")
        self.sprites["bg_night"] = self._load_sprite("background-night.png")
        
        # Ground/Base
        self.sprites["base"] = self._load_sprite("base.png")
        
        # Pipes
        pipe = self._load_sprite("pipe-green.png")
        self.sprites["pipe_bottom"] = pipe
        self.sprites["pipe_top"] = pygame.transform.flip(pipe, False, True)
        
        # Birds (all three colors with animation frames)
        self.sprites["birds"] = {}
        for color in ["yellow", "blue", "red"]:
            self.sprites["birds"][color] = [
                self._load_sprite(f"{color}bird-downflap.png"),
                self._load_sprite(f"{color}bird-midflap.png"),
                self._load_sprite(f"{color}bird-upflap.png"),
            ]
        
        # Number sprites for score display
        self.sprites["numbers"] = [
            self._load_sprite(f"{i}.png") for i in range(10)
        ]
        
        # UI sprites
        self.sprites["message"] = self._load_sprite("message.png")
        self.sprites["gameover"] = self._load_sprite("gameover.png")
        
        # Sounds
        self.sounds["die"] = self._load_sound("die.wav")
        self.sounds["hit"] = self._load_sound("hit.wav")
        self.sounds["point"] = self._load_sound("point.wav")
        self.sounds["swoosh"] = self._load_sound("swoosh.wav")
        self.sounds["wing"] = self._load_sound("wing.wav")
    
    def play_sound(self, name: str):
        """Play a sound by name if it exists."""
        sound = self.sounds.get(name)
        if sound:
            sound.play()
    
    def get_bird_sprite(self, color: str, frame: int) -> pygame.Surface:
        """Get bird sprite for given color and animation frame."""
        return self.sprites["birds"][color][frame % 3]
    
    @property
    def pipe_width(self) -> int:
        return self.sprites["pipe_bottom"].get_width()
    
    @property
    def pipe_height(self) -> int:
        return self.sprites["pipe_bottom"].get_height()
    
    @property
    def base_height(self) -> int:
        return self.sprites["base"].get_height()
    
    @property
    def bird_size(self) -> tuple[int, int]:
        """Returns (width, height) of bird sprite."""
        sprite = self.sprites["birds"]["yellow"][0]
        return sprite.get_width(), sprite.get_height()

class Brain:
    
    def __init__(self, weights: list[float] | None = None):
        if weights:
            self.weights = weights.copy()
        else:
            self.weights = [
                random.uniform(-5, 5),   # weight for distance to gap
                random.uniform(-5, 5),   # weight for velocity
                random.uniform(-1, 1),   # bias
            ]
    
    def should_flap(self, bird_y: float, velocity: float, gap_y: float) -> bool:
        """Decide whether the bird should flap based on current state."""
        # Normalize inputs
        distance_to_gap = (gap_y - bird_y) / Config.SCREEN_HEIGHT
        normalized_velocity = velocity / (10.0 * Config.SCALE)
        
        # Compute activation
        activation = (
            self.weights[0] * distance_to_gap +
            self.weights[1] * normalized_velocity +
            self.weights[2]  # bias
        )
        
        return activation > 0
    
    def copy(self) -> "Brain":
        """Create a copy of this brain."""
        return Brain(self.weights)
    
    @staticmethod
    def crossover(parent_a: "Brain", parent_b: "Brain") -> "Brain":
        """Create a child brain by mixing two parent brains."""
        child_weights = [
            wa if random.random() < 0.5 else wb
            for wa, wb in zip(parent_a.weights, parent_b.weights)
        ]
        return Brain(child_weights)
    
    def mutate(self) -> "Brain":
        """Return a mutated copy of this brain."""
        new_weights = [
            w + random.gauss(0, Config.MUTATION_STD)
            if random.random() < Config.MUTATION_RATE else w
            for w in self.weights
        ]
        return Brain(new_weights)


class Bird:
    """A single bird controlled by a neural network brain."""
    
    COLORS = ["yellow", "blue", "red"]
    
    def __init__(self, brain: Brain | None = None, color: str | None = None):
        self.y = Config.SCREEN_HEIGHT // 2
        self.velocity = 0.0
        self.brain = brain or Brain()
        self.color = color or random.choice(self.COLORS)
        self.alive = True
        self.score = 0.0
        self.flap_offset = random.randint(0, 20)  # Animation variety
    
    def update(self, gap_y: float, floor_y: float, pipe_x: float, 
               pipe_width: int, gap_size: int, bird_width: int, bird_height: int) -> bool:
        """
        Update bird physics and check for death.
        Returns True if bird flapped this frame.
        """
        if not self.alive:
            return False
        
        # Brain decides whether to flap
        flapped = False
        if self.brain.should_flap(self.y, self.velocity, gap_y):
            self.velocity = Config.FLAP_VELOCITY
            flapped = True
        
        # Apply physics
        self.velocity += Config.GRAVITY
        self.y += self.velocity
        
        # Check collisions
        if self._check_collision(floor_y, pipe_x, pipe_width, gap_y, gap_size, 
                                  bird_width, bird_height):
            self.alive = False
            return flapped
        
        # Survived this frame - increase fitness
        self.score += 0.1
        return flapped
    
    def _check_collision(self, floor_y: float, pipe_x: float, pipe_width: int,
                         gap_y: float, gap_size: int, bird_width: int, 
                         bird_height: int) -> bool:
        """Check if bird collides with ground, ceiling, or pipes."""
        half_width = bird_width // 2
        half_height = bird_height // 2
        
        # Ground collision
        if self.y + half_height >= floor_y:
            return True
        
        # Ceiling collision
        if self.y - half_height <= 0:
            return True
        
        # Pipe collision (only check if in pipe's x range)
        bird_left = Config.BIRD_X - half_width
        bird_right = Config.BIRD_X + half_width
        pipe_right = pipe_x + pipe_width
        
        if pipe_x < bird_right and pipe_right > bird_left:
            top_pipe_bottom = gap_y - gap_size // 2
            bottom_pipe_top = gap_y + gap_size // 2
            
            if self.y - half_height < top_pipe_bottom or self.y + half_height > bottom_pipe_top:
                return True
        
        return False
    
    def award_pipe_bonus(self):
        """Award bonus points for passing a pipe."""
        if self.alive:
            self.score += 5
    
    def get_sprite(self, assets: Assets, frame: int) -> pygame.Surface:
        """Get the current sprite with rotation based on velocity."""
        anim_frame = ((frame + self.flap_offset) // 5) % 3
        sprite = assets.get_bird_sprite(self.color, anim_frame)
        
        # Rotate based on velocity
        angle = max(-25, min(90, -self.velocity * 3))
        return pygame.transform.rotate(sprite, angle)


class Pipe:
    """A pair of pipes (top and bottom) that birds must navigate through."""
    
    def __init__(self, x: float | None = None):
        self.x = x if x is not None else Config.SCREEN_WIDTH + 50
        self.gap_y = self._random_gap_y()
    
    def _random_gap_y(self) -> int:
        """Generate a random gap position."""
        min_y = int(120 * Config.SCALE)
        max_y = int((Config.BASE_HEIGHT - 120) * Config.SCALE)
        return random.randint(min_y, max_y)
    
    def update(self) -> bool:
        """
        Move the pipe left.
        Returns True if pipe went off screen and was reset.
        """
        self.x -= Config.PIPE_SPEED
        
        if self.x + 100 < 0:  # Off screen
            self.reset()
            return True
        return False
    
    def reset(self):
        """Reset pipe to starting position with new gap."""
        self.x = Config.SCREEN_WIDTH + 50
        self.gap_y = self._random_gap_y()
    
    def check_passed(self, bird_x: float, pipe_width: int) -> bool:
        """Check if a bird at bird_x just passed this pipe."""
        old_right = self.x + Config.PIPE_SPEED + pipe_width
        new_right = self.x + pipe_width
        return old_right >= bird_x and new_right < bird_x
    
    def draw(self, screen: pygame.Surface, assets: Assets):
        """Draw the pipe pair."""
        pipe_width = assets.pipe_width
        pipe_height = assets.pipe_height
        
        # Top pipe (flipped)
        top_y = self.gap_y - Config.GAP_SIZE // 2 - pipe_height
        screen.blit(assets.sprites["pipe_top"], (int(self.x), top_y))
        
        # Bottom pipe
        bottom_y = self.gap_y + Config.GAP_SIZE // 2
        screen.blit(assets.sprites["pipe_bottom"], (int(self.x), bottom_y))


class Population:
    """Manages a population of birds and handles evolution."""
    
    def __init__(self):
        self.birds: list[Bird] = []
        self.generation = 1
        self.best_fitness_ever = 0.0  # Internal fitness for GA
        self.best_pipes_ever = 0      # Best pipes passed (for display)
        self._create_initial_population()
    
    def _create_initial_population(self):
        """Create the initial random population."""
        self.birds = []
        for i in range(Config.POPULATION_SIZE):
            color = Bird.COLORS[i % len(Bird.COLORS)]
            self.birds.append(Bird(color=color))
    
    @property
    def alive_count(self) -> int:
        """Count of birds still alive."""
        return sum(1 for bird in self.birds if bird.alive)
    
    @property
    def all_dead(self) -> bool:
        """Check if all birds are dead."""
        return self.alive_count == 0
    
    @property
    def alive_birds(self) -> list[Bird]:
        """Get list of alive birds."""
        return [bird for bird in self.birds if bird.alive]
    
    def evolve(self):
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness
        sorted_birds = sorted(self.birds, key=lambda b: b.score, reverse=True)
        best_fitness = sorted_birds[0].score if sorted_birds else 0.0
        self.best_fitness_ever = max(self.best_fitness_ever, best_fitness)
        
        # Select elites
        elite_count = max(1, int(Config.POPULATION_SIZE * Config.ELITE_FRACTION))
        elites = sorted_birds[:elite_count]
        
        # Create new population
        new_birds = []
        
        # Copy elites directly
        for elite in elites:
            new_birds.append(Bird(
                brain=elite.brain.copy(),
                color=elite.color
            ))
        
        # Fill rest with offspring
        while len(new_birds) < Config.POPULATION_SIZE:
            parent_a, parent_b = random.sample(elites, 2)
            child_brain = Brain.crossover(parent_a.brain, parent_b.brain).mutate()
            new_birds.append(Bird(
                brain=child_brain,
                color=random.choice(Bird.COLORS)
            ))
        
        self.birds = new_birds
        self.generation += 1
    
    def reset(self):
        """Reset population to initial state."""
        self.generation = 1
        self.best_fitness_ever = 0.0
        self.best_pipes_ever = 0
        self._create_initial_population()
    
    def update_best_pipes(self, pipes_passed: int):
        """Update the best pipes record if current is higher."""
        self.best_pipes_ever = max(self.best_pipes_ever, pipes_passed)


class Renderer:
    """Handles all drawing operations."""
    
    def __init__(self, screen: pygame.Surface, assets: Assets):
        self.screen = screen
        self.assets = assets
        self.font = pygame.font.SysFont(None, 32)
        self.font_small = pygame.font.SysFont(None, 24)
        self.base_x = 0  # For scrolling ground
        self.current_bg = "bg_day"
    
    def scroll_base(self):
        """Update base scroll position."""
        self.base_x -= Config.PIPE_SPEED
        base_width = self.assets.sprites["base"].get_width()
        if self.base_x <= -base_width:
            self.base_x = 0
    
    def toggle_background(self):
        """Randomly toggle between day/night background."""
        if random.random() < 0.2:
            self.current_bg = "bg_night" if self.current_bg == "bg_day" else "bg_day"
    
    def draw_background(self):
        """Draw the background."""
        self.screen.blit(self.assets.sprites[self.current_bg], (0, 0))
    
    def draw_base(self):
        """Draw the scrolling ground."""
        base = self.assets.sprites["base"]
        base_width = base.get_width()
        base_y = Config.SCREEN_HEIGHT - self.assets.base_height
        
        self.screen.blit(base, (self.base_x, base_y))
        self.screen.blit(base, (self.base_x + base_width, base_y))
    
    def draw_bird(self, bird: Bird, frame: int):
        """Draw a single bird."""
        if not bird.alive:
            return
        
        sprite = bird.get_sprite(self.assets, frame)
        x = Config.BIRD_X - sprite.get_width() // 2
        y = int(bird.y) - sprite.get_height() // 2
        self.screen.blit(sprite, (x, y))
    
    def draw_score(self, score: int, y_offset: int = 0):
        """Draw score using number sprites."""
        score_str = str(score)
        numbers = self.assets.sprites["numbers"]
        
        total_width = sum(numbers[int(d)].get_width() for d in score_str)
        x = (Config.SCREEN_WIDTH - total_width) // 2
        y = int(30 * Config.SCALE) + y_offset
        
        for digit in score_str:
            digit_sprite = numbers[int(digit)]
            self.screen.blit(digit_sprite, (x, y))
            x += digit_sprite.get_width()
    
    def draw_hud(self, generation: int, alive_count: int, best_pipes: int):
        """Draw the heads-up display."""
        text = f"Gen: {generation}  Alive: {alive_count}  Best: {best_pipes}"
        
        # Shadow
        shadow = self.font_small.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (12, Config.SCREEN_HEIGHT - 28))
        
        # Text
        surface = self.font_small.render(text, True, (255, 255, 255))
        self.screen.blit(surface, (10, Config.SCREEN_HEIGHT - 30))
    
    def draw_nerd_mode_indicator(self):
        """Draw nerd mode label and legend."""
        # Label
        label = self.font_small.render("NERD MODE [N]", True, (255, 50, 50))
        self.screen.blit(label, (10, int(60 * Config.SCALE)))
        
        # Legend
        legend_y = Config.SCREEN_HEIGHT - 100
        
        pygame.draw.line(self.screen, (255, 255, 0), (10, legend_y), (30, legend_y), 1)
        text1 = self.font_small.render("= sight to gap", True, (255, 255, 255))
        self.screen.blit(text1, (35, legend_y - 8))
        
        pygame.draw.line(self.screen, (0, 255, 0), (10, legend_y + 20), (30, legend_y + 20), 2)
        text2 = self.font_small.render("= velocity", True, (255, 255, 255))
        self.screen.blit(text2, (35, legend_y + 12))
    
    def draw_nerd_mode_hint(self):
        """Draw hint to enable nerd mode."""
        shadow = self.font_small.render("Press N for nerd mode", True, (0, 0, 0))
        text = self.font_small.render("Press N for nerd mode", True, (255, 255, 255))
        y = int(60 * Config.SCALE)
        self.screen.blit(shadow, (12, y + 2))
        self.screen.blit(text, (10, y))
    
    def draw_nerd_visuals(self, bird: Bird, gap_y: float, pipe_x: float, pipe_width: int):
        """Draw AI visualization for a bird."""
        if not bird.alive:
            return
        
        bird_y = int(bird.y)
        gap_center_x = int(pipe_x + pipe_width // 2)
        
        # Sight line to gap
        pygame.draw.line(
            self.screen, (255, 255, 0),
            (Config.BIRD_X, bird_y),
            (gap_center_x, int(gap_y)), 1
        )
        
        # Velocity vector
        vel_end_y = bird_y + bird.velocity * 3
        pygame.draw.line(
            self.screen, (0, 255, 0),
            (Config.BIRD_X, bird_y),
            (Config.BIRD_X + int(15 * Config.SCALE), int(vel_end_y)), 2
        )
        
        # Activation bar
        distance_to_gap = (gap_y - bird.y) / Config.SCREEN_HEIGHT
        normalized_vel = bird.velocity / (10.0 * Config.SCALE)
        activation = (
            bird.brain.weights[0] * distance_to_gap +
            bird.brain.weights[1] * normalized_vel +
            bird.brain.weights[2]
        )
        
        bar_height = min(abs(activation) * 10, 30)
        bar_color = (255, 100, 100) if activation > 0 else (100, 100, 255)
        bar_y = bird_y - bar_height if activation > 0 else bird_y
        
        pygame.draw.rect(
            self.screen, bar_color,
            (Config.BIRD_X + int(10 * Config.SCALE), int(bar_y), 4, int(bar_height))
        )
    
    def draw_menu_screen(self, frame: int):
        """Draw the start menu screen."""
        self.draw_background()
        
        # Animated birds
        t = pygame.time.get_ticks() / 1000
        for i, color in enumerate(Bird.COLORS):
            x = int(60 * Config.SCALE) + i * int(90 * Config.SCALE)
            y = int(180 * Config.SCALE) + math.sin(t * 2 + i) * int(15 * Config.SCALE)
            anim_frame = (int(t * 10) + i * 3) % 3
            sprite = self.assets.get_bird_sprite(color, anim_frame)
            self.screen.blit(sprite, (int(x) - sprite.get_width() // 2, 
                                       int(y) - sprite.get_height() // 2))
        
        # Message sprite
        message = self.assets.sprites["message"]
        x = (Config.SCREEN_WIDTH - message.get_width()) // 2
        self.screen.blit(message, (x, int(100 * Config.SCALE)))
        
        # Draw base
        self.draw_base()
        
        # Pulsing start text
        alpha = int(128 + 127 * math.sin(t * 4))
        start_text = self.font.render("Press SPACE to start", True, (255, 255, 255))
        start_text.set_alpha(alpha)
        self.screen.blit(start_text, 
                         (Config.SCREEN_WIDTH // 2 - start_text.get_width() // 2, 
                          int(420 * Config.SCALE)))
        
        # Controls
        controls = ["N - Toggle Nerd Mode", "R - Restart"]
        for i, line in enumerate(controls):
            text = self.font_small.render(line, True, (50, 50, 50))
            self.screen.blit(text, 
                            (Config.SCREEN_WIDTH // 2 - text.get_width() // 2, 
                             int(450 * Config.SCALE) + i * 20))


class Game:
    """Main game class that coordinates all game systems."""
    
    # Game states
    STATE_MENU = 0
    STATE_PLAYING = 1
    
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        
        self.screen = pygame.display.set_mode(
            (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("AI Flappy Bird - Neuroevolution")
        
        self.clock = pygame.time.Clock()
        self.assets = Assets()
        self.renderer = Renderer(self.screen, self.assets)
        
        self.state = self.STATE_MENU
        self.nerd_mode = False
        self.frame = 0
        
        self.population = Population()
        self.pipe = Pipe()
        self.pipes_passed = 0
        
        # Compute floor Y position
        self.floor_y = Config.SCREEN_HEIGHT - self.assets.base_height + int(20 * Config.SCALE)
    
    def handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                if event.key == pygame.K_SPACE and self.state == self.STATE_MENU:
                    self._start_game()
                
                if event.key == pygame.K_n:
                    self.nerd_mode = not self.nerd_mode
                
                if event.key == pygame.K_r:
                    self._restart()
        
        return True
    
    def _start_game(self):
        """Start a new game."""
        self.state = self.STATE_PLAYING
        self.population.reset()
        self.pipe.reset()
        self.pipes_passed = 0
        self.assets.play_sound("swoosh")
    
    def _restart(self):
        """Restart the simulation."""
        self.population.reset()
        self.pipe.reset()
        self.pipes_passed = 0
        self.state = self.STATE_PLAYING
        self.assets.play_sound("swoosh")
    
    def update(self):
        """Update game state."""
        if self.state != self.STATE_PLAYING:
            return
        
        self.frame += 1
        
        # Update pipe
        if self.pipe.update():
            self.renderer.toggle_background()
        
        # Check if birds passed the pipe
        if self.pipe.check_passed(Config.BIRD_X, self.assets.pipe_width):
            self.pipes_passed += 1
            self.assets.play_sound("point")
            for bird in self.population.alive_birds:
                bird.award_pipe_bonus()
        
        # Update all birds
        bird_width, bird_height = self.assets.bird_size
        any_flapped = False
        any_died = False
        
        for bird in self.population.birds:
            if not bird.alive:
                continue
            
            was_alive = bird.alive
            flapped = bird.update(
                gap_y=self.pipe.gap_y,
                floor_y=self.floor_y,
                pipe_x=self.pipe.x,
                pipe_width=self.assets.pipe_width,
                gap_size=Config.GAP_SIZE,
                bird_width=bird_width,
                bird_height=bird_height
            )
            
            if flapped:
                any_flapped = True
            
            if was_alive and not bird.alive:
                any_died = True
        
        # Sound effects
        if any_flapped and random.random() < 0.1:
            self.assets.play_sound("wing")
        
        if any_died:
            self.assets.play_sound("hit")
        
        # Scroll ground
        self.renderer.scroll_base()
        
        # Check for extinction - evolve next generation
        if self.population.all_dead:
            self.assets.play_sound("die")
            self.population.update_best_pipes(self.pipes_passed)  # Track best before reset
            self.population.evolve()
            self.pipe.reset()
            self.pipes_passed = 0
    
    def draw(self):
        """Render the current frame."""
        if self.state == self.STATE_MENU:
            self.renderer.draw_menu_screen(self.frame)
        else:
            self._draw_game()
        
        pygame.display.flip()
    
    def _draw_game(self):
        """Draw the gameplay screen."""
        # Background
        self.renderer.draw_background()
        
        # Pipe
        self.pipe.draw(self.screen, self.assets)
        
        # Birds with optional nerd visuals
        for bird in self.population.birds:
            if self.nerd_mode and bird.alive:
                self.renderer.draw_nerd_visuals(
                    bird, self.pipe.gap_y, self.pipe.x, self.assets.pipe_width
                )
            self.renderer.draw_bird(bird, self.frame)
        
        # Ground (drawn on top)
        self.renderer.draw_base()
        
        # Score
        self.renderer.draw_score(self.pipes_passed)
        
        # HUD
        self.renderer.draw_hud(
            self.population.generation,
            self.population.alive_count,
            self.population.best_pipes_ever
        )
        
        # Nerd mode UI
        if self.nerd_mode:
            self.renderer.draw_nerd_mode_indicator()
        else:
            self.renderer.draw_nerd_mode_hint()
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            self.clock.tick(60)
            running = self.handle_events()
            self.update()
            self.draw()
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()
