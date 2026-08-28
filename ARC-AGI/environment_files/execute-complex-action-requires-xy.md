> ## Documentation Index
> Fetch the complete documentation index at: https://docs.arcprize.org/llms.txt
> Use this file to discover all available pages before exploring further.

# Execute complex action (requires x,y)

> Issues **ACTION 6**—a two-parameter command that supplies explicit
X/Y coordinates—to an active game session.  Common use-cases
include “click/tap at (x,y)”, “place a tile”, or “shoot a
projectile,” depending on the game's mechanics.

Required fields  
• `game_id` — the game to act in  
• `guid`    — session identifier obtained from RESET  
• `x`,`y`   — zero-based grid coordinates (0-63 inclusive)

On success the server applies the action, advances game logic to
the next stable frame, and returns that frame together with the
updated score, state, and win condition.




> ## OpenAPI

````yaml /arc3v1.yaml post /api/cmd/ACTION6
openapi: 3.0.3
info:
  title: ARC‑AGI‑3 REST API
  version: 1.0.0
  description: >
    Programmatic interface for running agents against ARC‑AGI‑3 games,
    opening/closing score‑cards and driving game state with actions.

    All requests **require** an `X‑API‑Key` header issued from the ARC‑AGI‑3 web
    console.


    **Important: Session Affinity via Cookies**  

    Games are stateful and require session affinity. The server sets cookies
    (especially `AWSALB*` cookies) in responses that **must be included in all
    subsequent requests** for the same game session. These cookies route
    requests to the correct backend instance maintaining your game state. Most
    HTTP clients handle cookies automatically, but ensure your client preserves
    and sends cookies received from RESET and ACTION responses.
servers:
  - url: https://three.arcprize.org
security: []
tags:
  - name: Games
  - name: Scorecards
  - name: Commands
paths:
  /api/cmd/ACTION6:
    post:
      tags:
        - Commands
      summary: Execute complex action (requires x,y)
      description: |
        Issues **ACTION 6**—a two-parameter command that supplies explicit
        X/Y coordinates—to an active game session.  Common use-cases
        include “click/tap at (x,y)”, “place a tile”, or “shoot a
        projectile,” depending on the game's mechanics.

        Required fields  
        • `game_id` — the game to act in  
        • `guid`    — session identifier obtained from RESET  
        • `x`,`y`   — zero-based grid coordinates (0-63 inclusive)

        On success the server applies the action, advances game logic to
        the next stable frame, and returns that frame together with the
        updated score, state, and win condition.
      operationId: action6
      requestBody:
        description: Game/session identifiers plus the coordinate payload.
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ComplexActionCommand'
            examples:
              placeTile:
                summary: Place at (12, 34)
                value:
                  game_id: ls20-016295f7601e
                  guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                  x: 12
                  'y': 34
      responses:
        '200':
          description: Frame returned after executing the action.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FrameResponse'
              examples:
                frame:
                  value:
                    game_id: ls20-016295f7601e
                    guid: 2fa5332c-2e55-4825-b5c5-df960d504470
                    frame:
                      - - - …
                    state: NOT_FINISHED
                    levels_completed: 17
                    win_levels: 254
                    action_input:
                      id: 6
                      data:
                        x: 12
                        'y': 34
                    available_actions:
                      - 1
                      - 2
                      - 3
                      - 4
        '400':
          description: |
            Bad request - possible causes:  
            • Unknown `game_id`  
            • `guid` not found or does not belong to the supplied `game_id`  
            • `x` or `y` outside the 0-63 range
        '401':
          description: Missing or invalid **X-API-Key** header.
      security:
        - ApiKeyAuth: []
components:
  schemas:
    ComplexActionCommand:
      type: object
      description: >
        Payload for coordinate-based actions (e.g. `/api/cmd/ACTION6`).

        Supplies an `(x, y)` location on the 64 × 64 game grid along with

        the game/session identifiers so the engine can apply the action

        to the correct running instance.


        **Important:** Include any cookies (especially `AWSALB*` cookies)
        received from previous RESET or ACTION responses to ensure session
        affinity.
      properties:
        game_id:
          type: string
          description: Identifier of the game receiving this action.
        guid:
          type: string
          description: Server-generated session ID obtained from the RESET call.
        x:
          type: integer
          minimum: 0
          maximum: 63
          description: Horizontal coordinate on the game grid (0 = left, 63 = right).
        'y':
          type: integer
          minimum: 0
          maximum: 63
          description: Vertical coordinate on the game grid (0 = top, 63 = bottom).
        reasoning:
          type: object
          description: |
            Optional, caller-defined JSON blob (≤ 16 KB) capturing the
            agent's internal reasoning, model parameters, or any other
            metadata you'd like to store alongside the action.
          additionalProperties: true
      required:
        - game_id
        - guid
        - x
        - 'y'
    FrameResponse:
      type: object
      description: |
        Snapshot returned after every RESET or ACTION command.  
        Includes the latest visual frame(s), cumulative score details, the
        current game state, and an echo of the triggering action.
      properties:
        game_id:
          type: string
          description: Game identifier for the running session.
        guid:
          type: string
          description: Server-generated session ID; use this for all subsequent commands.
        frame:
          type: array
          description: |
            One or more consecutive visual frames. Each frame is a 64 × 64
            grid of 4-bit colour indices (integers 0-15). Multiple frames
            may be returned if the environment advances internally (e.g.,
            animations) before settling.
          items:
            type: array
            items:
              type: array
              items:
                type: integer
                minimum: 0
                maximum: 15
        state:
          type: string
          description: >
            Current state of the session:


            • **NOT_FINISHED** - game in progress, not yet WIN or GAME_OVER.  

            • **NOT_STARTED**  - session has ended (WIN or GAME_OVER) and
            requires RESET.  

            • **WIN**          - session ended in victory.  

            • **GAME_OVER**    - session ended in defeat.
          enum:
            - NOT_FINISHED
            - NOT_STARTED
            - WIN
            - GAME_OVER
        levels_completed:
          type: integer
          description: Current cumulative number of levels completed for this run.
          minimum: 0
          maximum: 254
        win_levels:
          type: integer
          description: |
            Level threshold required to reach the **WIN** state. Mirrors
            the game's configured win condition so agents can adapt
            dynamically without hard-coding values.
          minimum: 0
          maximum: 254
        action_input:
          type: object
          description: Echo of the command that produced this frame.
          properties:
            id:
              type: integer
              description: Client-assigned or sequential action index.
            data:
              type: object
              description: Additional parameters originally sent with the action.
              additionalProperties: true
        available_actions:
          type: array
          description: List of available actions for the current game.
          items:
            type: integer
            enum:
              - 1
              - 2
              - 3
              - 4
              - 5
              - 6
      required:
        - game_id
        - guid
        - frame
        - state
        - levels_completed
        - win_levels
        - action_input
        - available_actions
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

````
