import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction

env = mill.env(render_mode='human')

# Wrap the Mill environment into the UserInteraction wrapper
# to enable the user to interact with the board.
env = UserInteraction(env)
env.reset()

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
    if agent == "player_1":

        # Mark all positions to which the player can move a piece.
        for [_, dst, _] in info["legal_moves"]:
            env.mark_position(dst, (128, 128, 0, 128))  # yellow shade.

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

            # If the user quit, let us truncate the game.
            if event["type"] == "quit":
                done_interacting = True
                truncation = True
            
            # Use a different selection color for empty and occupied positions.
            elif info["phase"] == "placing" and event["type"] == "mouse_move":
                if observation[event["position"] - 1] == 0:
                    env.set_selection_color((64, 192, 0, 128))  # Green shade.
                else:
                    env.set_selection_color((128, 128, 255, 255))  # Blue shade.
            # On flying and moving phase we just color the pieces of the player
            elif event["type"] == "mouse_move":
                 for position in initial_model._all_pieces(1):
                    # We color only the current pieces that can be moved to other position
                    if any(src == position for [src, dst, capture] in info["legal_moves"]):
                        env.mark_position(position, (128, 128, 255, 128)) # Light blue shade
            
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
                # On moving and flying phase we also need to specify the piece we move and where we move it
                else:
                    if observation[event["position"] - 1] == 1:
                        env.mark_position(position, (128, 128, 255, 256)) # Darker blue as it is the one selected
                        while True:  
                        
                            # Mark all the possible moves in green.
                            posible_moves = []
                            for [src, dst, capture] in info["legal_moves"]:
                                 if src == event["position"]:
                                      posible_moves.append([src, dst, capture])
                                      env.mark_position(dst, (128, 128, 0, 128))

                            # If there is just one move we don't need to give the user the option to choose
                            if len(posible_moves) == 1:
                                move = posible_moves[0]
                                done_interacting = True
                                canCapture = True if posible_moves[0][2] > 0 else False
                                break
                
                            event = env.interact()

                            # If the user quit, let us truncate the game.
                            if event["type"] == "quit":
                                            done_interacting = True
                                            truncation = True
                                            break
                
                            # If the player chooses an place to move his piece that is posible then we move it
                            if event["type"] == "mouse_click":
                                for [src, dst, capture] in posible_moves:
                                    if dst == event["position"]:
                                        move = [src, dst, capture]
                                        done_interacting = True
                                        canCapture = True if posible_moves[2] > 0 else False
                                        break
                            # We also paint a darker green into the "selected" free space
                            elif event["type"] == "mouse_move":
                                if observation[event["position"] - 1] == 0:
                                    env.set_selection_color((64, 192, 0, 128))  # Green shade.
                                
                
                            # Use a red shade for the pieces that can be captured:
                            print("Elige una posición para moverte")
                
                # If the player can do a capture then we have to let him choose the piece from the enemy he wants to capture:
                if(canCapture):                               
                    posible_capture = initial_model._capture_pieces(2)
                    while True:
            
            
                        # Mark all the possible captured pieces in red.
                        for position in posible_capture:
                            env.mark_position(position, (128, 0, 0, 128))
            
                        event = env.interact()

                        # If the user quit, let us truncate the game.
                        if event["type"] == "quit":
                                        done_interacting = True
                                        truncation = True
            
                        # If the player chooses a capturable piece then he captures that piece
                        if event["type"] == "mouse_click" and any(position == event["position"] for position in posible_capture):
                            move[2] = event["position"]
                            break
                        # We also paint a darker shadow into the "selected" piece
                        elif event["type"] == "mouse_move":
                            if observation[event["position"] - 1] == 2:
                                env.set_selection_color((255, 0, 0, 255))  # Darker Red.
                            
            
                        # Use a red shade for the pieces that can be captured:
                        print("Elige una pieza para capturar")
                break     

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

    # Make the chosen move.
    env.step(move)

# Close the environment.
env.close()