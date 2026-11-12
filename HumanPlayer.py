from famnit_gym.envs import mill


class HumanPlayer:
    def __init__(self, player, env):
        if player not in [1, 2]:
            raise ValueError(f"Unsupported player index: {player}")

        self.player = player
        self.env = env
        self.quit_requested = False

    def play(self):
        observation, _, _, _, info = self.env.last()
        opponent = 1 if self.player == 2 else 2
        # We need the model to calculate the possible pieces to capture.
        initial_model = mill.transition_model(self.env.unwrapped)
        move = None

        # Mark all positions to which the player can move a piece.
        if info["phase"] == "placing":
            for [_, dst, _] in info["legal_moves"]:
                self.env.mark_position(dst, (128, 128, 0, 128))
        # Mark all the position where the player has a piece he can move.
        else:
            for [src, _, _] in info["legal_moves"]:
                self.env.mark_position(src, (128, 128, 255, 128))

        # Here, the user interacts with the board for a while.
        done_interacting = False
        while not done_interacting:
            # Blocks, until one of the following events occurs:
            # - The user quits (closes the window).
            # - A mouse is move off a position (0) or onto a new position (1 - 24).
            # - A mouse button has been clicked.
            # - A key has been pressed.
            event = self.env.interact()

            # We reset the capture flag 
            can_capture = False

            # We reset the reset move flag
            reset_move = False

            # If the user quit, let us truncate the game.
            if event["type"] == "quit":
                done_interacting = True
                self.quit_requested = True

            # Use a different selection color for empty and occupied positions.
            elif event["type"] == "mouse_move":
                if observation[event["position"] - 1] == 0:
                    self.env.set_selection_color((64, 192, 0, 128))
                else:
                    self.env.set_selection_color((128, 128, 255, 255))

            # Only if clicked on an empty position, the game continues.
            elif event["type"] == "mouse_click":
                # If we are in placing phase we just need to select the position where to place
                if info["phase"] == "placing":
                    if observation[event["position"] - 1] == 0:
                        # Check if there is a legal move to move there.
                        for [src, dst, capture] in info["legal_moves"]:
                            if dst == event["position"]:
                                # In this phase we never do captures
                                move = [src, dst, 0]
                                done_interacting = True
                                can_capture = capture > 0
                                break
                                
                # On moving and flying phase we also need to specify the piece we move and where we move it
                elif observation[event["position"] - 1] == self.player:
                    position = event["position"]
                    self.env.mark_position(position, (0, 0, 255, 255)) # Darker blue as it is the one selected
                    while True:
                        # Mark all the possible moves in green.
                        posible_moves = []
                        for [src, dst, capture] in info["legal_moves"]:
                            if src == event["position"]:
                                posible_moves.append([src, dst, capture])
                                self.env.mark_position(dst, (128, 128, 0, 128)) # Yellow shade

                        # If there is just one move we don't need to give the user the option to choose
                        if len(posible_moves) == 1:
                            move = posible_moves[0]
                            done_interacting = True
                            can_capture = True if posible_moves[0][2] > 0 else False
                            break
            
                        sub_event = self.env.interact()

                        # If the user quit, let us truncate the game.
                        if sub_event["type"] == "quit":
                            done_interacting = True
                            self.quit_requested = True
                            break

                        # If the player chooses an place to move his piece that is posible then we move it
                        elif sub_event["type"] == "mouse_click":

                            # If we select back again the same piece we undo the move 
                            if sub_event["position"] == event["position"]:
                                reset_move = True
                                done_interacting = False
                                break

                            for [src, dst, capture] in posible_moves:
                                if dst == sub_event["position"]:
                                    move = [src, dst, capture]
                                    done_interacting = True
                                    can_capture = True if capture > 0 else False
                                    break
                            break
                        # We also paint a darker green into the "selected" free space
                        elif sub_event["type"] == "mouse_move":
                            if observation[sub_event["position"] - 1] == 0:
                                self.env.set_selection_color((64, 192, 0, 128))  # Green shade.
                            
            
                # If the player can do a capture then we have to let him choose
                # the piece from the enemy he wants to capture.
                if(can_capture):
                    posible_capture = initial_model._capture_pieces(opponent)
                    while True:
            
                        # Mark all the possible captured pieces in red.
                        for position in posible_capture:
                            self.env.mark_position(position, (128, 0, 0, 128))
            
                        sub_event = self.env.interact()

                        # If we click the piece again we can undo the move while we see the capture map
                        if sub_event["type"] == "mouse_click" and sub_event["position"] == event["position"]:
                            done_interacting = False
                            can_capture = False
                            reset_move = True
                            break

                        # If the user quit, let us truncate the game.
                        if sub_event["type"] == "quit":
                            done_interacting = True
                            self.quit_requested = True
                            break
            
                        # If the player chooses a capturable piece then he captures that piece
                        if sub_event["type"] == "mouse_click" and any(position == sub_event["position"] for position in posible_capture):
                            move[2] = sub_event["position"]
                            break

                        # We also paint a darker shadow into the "selected" piece
                        elif sub_event["type"] == "mouse_move":
                            if observation[sub_event["position"] - 1] == opponent:
                                self.env.set_selection_color((255, 0, 0, 255))  # Darker Red. 
                
                if reset_move:
                    # Clean everything
                    self.env.clear_markings()

                    # Paint everything as default
                    if info["phase"] == "placing":
                        for [_, dst, _] in info["legal_moves"]:
                            self.env.mark_position(dst, (128, 128, 0, 128))  # yellow shade.
                                        
                    # Mark all the position where the player has a piece he can move
                    else:
                        for [src, _, _] in info["legal_moves"]:
                            self.env.mark_position(src, (128, 128, 255, 128))
                else: # We only escape the extern while loop if we didn't reset the move
                    break

            # If escape key has been pressed, the game is truncated.
            elif event["type"] == "key_press" and event["key"] == "escape":
                done_interacting = True
                self.quit_requested = True

        # Now clear all the markings.
        self.env.clear_markings()

        # If truncated during the interaction, the user quit interactively.
        if self.quit_requested:
            print("User quit interactively!")
            

        return move
