import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction
import famnit_gym.envs.mill.mill_model as MillModel
from agent import Agent
import time

env = mill.env(render_mode='human')

# Wrap the Mill environment into the UserInteraction wrapper
# to enable the user to interact with the board.
env = UserInteraction(env)
env.reset()
USER_PLAYING = True
USER_TURN = 1
IA_PLAYING = False

if USER_PLAYING:
    IA_TURN = 1 if USER_TURN == 2 else 2
else:
    IA_TURN = 1
True

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game is too long!")
        break

    # Let the user decide on the next move.
    move = None

    # We need the model to
    initial_model = mill.transition_model(env.unwrapped)


    # We choose between the player1 and the IA
    if agent == f"player_{USER_TURN}" and USER_PLAYING:
        # Mark all positions to which the player can move a piece.
        if info["phase"] == "placing":
            for [_, dst, _] in info["legal_moves"]:
                env.mark_position(dst, (128, 128, 0, 128))  # yellow shade.
        # Mark all the position where the player has a piece he can move
        else:
            for [src, _, _] in info["legal_moves"]:
                env.mark_position(src, (128, 128, 255, 128))  # blue shade.  
        # Here, the user interacts with the board for a while.
        # In this example, the interaction is done when the user
        # clicks on an empty position on the board.
        done_interacting = False
        while not done_interacting:
            # Blocks, until one of the following events occurs:
            # - The user quits (closes the window).
            # - A mouse is move off a position (0) or onto a new position (1 - 24).
            # - A mouse button has been clicked.
            # - A key has been pressed.
            event = env.interact()

            # We reset the capture flag 
            canCapture = False

            # We reset the reset move flag
            reset_move = False

            # If the user quit, let us truncate the game.
            if event["type"] == "quit":
                done_interacting = True
                truncation = True
            
            # Use a different selection color for empty and occupied positions.
            elif event["type"] == "mouse_move":
                if observation[event["position"] - 1] == 0:
                    env.set_selection_color((64, 192, 0, 128))  # Green shade.
                else:
                    env.set_selection_color((128, 128, 255, 255))  # Blue shade.
            
            # Only if clicked on an empty position, the game continues.
            elif event["type"] == "mouse_click":
                canCapture = False
                # If we are in placing phase we just need to select the position where to place
                if info["phase"] == "placing":
                    if observation[event["position"] - 1] == 0:
                        # Check if there is a legal move to move there.
                        for [src, dst, capture] in info["legal_moves"]:
                            # If we are on the pacing phase just move the piece if it is simply legal
                            if dst == event["position"]:
                                # In this phase we never do captures
                                move = [src, dst, 0]
                                done_interacting = True
                                canCapture = True if capture > 0 else False
                # On moving and flying phase we also need to specify the piece we move                        and where we move it
                else:
                    if observation[event["position"] - 1] == USER_TURN:
                        position = event["position"]
                        env.mark_position(position, (0, 0, 255, 255)) # Darker blue as it is the one selected
                        while True:  
                        
                            # Mark all the possible moves in green.
                            posible_moves = []
                            for [src, dst, capture] in info["legal_moves"]:
                                if src == event["position"]:
                                    posible_moves.append([src, dst, capture])
                                    env.mark_position(dst, (128, 128, 0, 128)) # Yellow shade

                            # If there is just one move we don't need to give the user the option to choose
                            if len(posible_moves) == 1:
                                move = posible_moves[0]
                                done_interacting = True
                                canCapture = True if posible_moves[0][2] > 0 else False
                                break
                
                            subEvent = env.interact()

                            # If the user quit, let us truncate the game.
                            if subEvent["type"] == "quit":
                                done_interacting = True
                                truncation = True
                                break

                            # If the player chooses an place to move his piece that is posible then we move it
                            elif subEvent["type"] == "mouse_click":
                                for [src, dst, capture] in posible_moves:
                                    if dst == subEvent["position"]:
                                        move = [src, dst, capture]
                                        done_interacting = True
                                        canCapture = True if capture > 0 else False
                                        break
                                break
                            # We also paint a darker green into the "selected" free space
                            elif subEvent["type"] == "mouse_move":
                                if observation[subEvent["position"] - 1] == 0:
                                    env.set_selection_color((64, 192, 0, 128))  # Green shade.
                                
                
                # If the player can do a capture then we have to let him choose the piece from the enemy he wants to capture:
                if(canCapture):                               
                    posible_capture = initial_model._capture_pieces(IA_TURN)
                    while True:
            
                        # Mark all the possible captured pieces in red.
                        for position in posible_capture:
                            env.mark_position(position, (128, 0, 0, 128))
            
                        subEvent = env.interact()

                        # If we click the piece again we can undo the move while we see the capture map
                        if subEvent["type"] == "mouse_click" and subEvent["position"] == event["position"]:
                            done_interacting = False
                            can_capture = False
                            reset_move = True
                            break

                        # If the user quit, let us truncate the game.
                        if subEvent["type"] == "quit":
                            done_interacting = True
                            truncation = True
                            break
            
                        # If the player chooses a capturable piece then he captures that piece
                        if subEvent["type"] == "mouse_click" and any(position == subEvent["position"] for position in posible_capture):
                            move[2] = subEvent["position"]
                            break

                        # We also paint a darker shadow into the "selected" piece
                        elif subEvent["type"] == "mouse_move":
                            if observation[subEvent["position"] - 1] == IA_TURN:
                                env.set_selection_color((255, 0, 0, 255))  # Darker Red. 
                if not reset_move: # We only escape the extern while loop if we didn't reset the move
                    break
                else:
                    # Clean everything
                    env.clear_markings()
                    # Paint everything as default
                    if info["phase"] == "placing":
                        for [_, dst, _] in info["legal_moves"]:
                            env.mark_position(dst, (128, 128, 0, 128))  # yellow shade.
                    # Mark all the position where the player has a piece he can move
                    else:
                        for [src, _, _] in info["legal_moves"]:
                            env.mark_position(src, (128, 128, 255, 128))  # blue shade.

            # If escape key has been pressed, the game is truncated.
            elif event["type"] == "key_press":
                if event["key"] == "escape":
                    done_interacting = True
                    truncation = True
        
        # Now clear all the markings.
        env.clear_markings()

        # Id truncated during the interaction, the user quit interactively.
        if truncation:
            print("User quit interactively!")
            break
    # The IA starts with a random move
    elif agent == f"player_{IA_TURN}" and IA_PLAYING:
        start = time.clock_gettime_ns(time.CLOCK_REALTIME)
        agent = Agent(IA_TURN, "hard")
        move = agent.play(original_model=initial_model)
        end = time.clock_gettime_ns(time.CLOCK_REALTIME)
        print(f"Tiempo empleado por montecarlo: {(end - start)/1_000_000}ms")
    # Dumb IA
    else:
        env.step(None)

    # Make the chosen move.
    if(move is not None):
        env.step(move)
    

# Close the environment.
env.close()