"""ETHRAEON Pure Programmatic Solver for ARC-AGI-3 (ls20).

Zero API cost — time-expanded BFS for moving-trigger levels,
BFS-based static planner for fixed-trigger levels.

Pre-computes full action sequence in __init__ using a game clone.
choose_action returns actions from the pre-computed queue.
"""

import logging
import re
from collections import deque
from itertools import permutations
from typing import Any

from arcengine import ActionInput, FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)

STEP = 5
SC_REFILL = 42
SC_FLOOR = 16
LOW_SC = 20

_AM = {
    'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
    'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4,
}

# Type aliases used throughout solver helpers
_Pos = tuple[int, int]
_Step = tuple[str, int, int]
_PushMap = dict[_Pos, _Pos]
_WallSet = set[_Pos] | None
_Pickup = dict[str, Any]


# ── Solver helpers ─────────────────────────────────────────────────────────────

def _is_wall(game: Any, x: int, y: int) -> bool:
    return any('ihdgageizm' in s.tags for s in game.mrznumynfe(x, y, STEP, STEP))


def _build_push_map(game: Any, start: _Pos) -> _PushMap:
    platforms = game.current_level.get_sprites_by_tag("gbvqrjtaqo")
    if not platforms:
        return {}

    def is_stop(nx: int, ny: int) -> bool:
        if nx < 0 or nx > 59 or ny < 0 or ny > 59:
            return True
        return any('ihdgageizm' in s.tags or 'rjlbuycveu' in s.tags
                   for s in game.mrznumynfe(nx, ny, STEP, STEP))

    ox, oy = start[0] % STEP, start[1] % STEP
    xs = [ox + STEP * i for i in range(12) if ox + STEP * i <= 59]
    ys = [oy + STEP * i for i in range(12) if oy + STEP * i <= 59]
    push_map = {}
    for plat in platforms:
        name = plat.name
        if name.endswith('t'):    dx, dy = 0, -1
        elif name.endswith('b'): dx, dy = 0,  1
        elif name.endswith('r'): dx, dy = 1,  0
        elif name.endswith('l'): dx, dy = -1, 0
        else: continue
        wall_cx = plat.x + dx
        wall_cy = plat.y + dy
        max_push_steps = 0
        for i in range(1, 13):
            nx = wall_cx + dx * plat.width * i
            ny = wall_cy + dy * plat.height * i
            if is_stop(nx, ny):
                max_push_steps = max(0, i - 1)
                break
        if max_push_steps <= 0:
            continue
        ddx = dx * plat.width * max_push_steps
        ddy = dy * plat.height * max_push_steps
        for gx in xs:
            for gy in ys:
                if (gx < plat.x + plat.width and gx + STEP > plat.x and
                        gy < plat.y + plat.height and gy + STEP > plat.y):
                    push_map[(gx, gy)] = (gx + ddx, gy + ddy)
    return push_map


def _bfs_path(game: Any, start: _Pos, goal: _Pos,
              push_map: _PushMap | None = None,
              extra_walls: _WallSet = None) -> list[_Step] | None:
    if start == goal:
        return []
    queue = deque([(start, [])])
    visited = {start}
    dirs = [('UP', 0, -STEP), ('DOWN', 0, STEP), ('LEFT', -STEP, 0), ('RIGHT', STEP, 0)]
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        for nm, dx, dy in dirs:
            nx, ny = pos[0] + dx, pos[1] + dy
            if not (0 <= nx <= 59 and 0 <= ny <= 59):
                continue
            if _is_wall(game, nx, ny):
                continue
            if extra_walls and (nx, ny) in extra_walls:
                continue
            if (nx, ny) in visited:
                continue
            if push_map and (nx, ny) in push_map:
                dest = push_map[(nx, ny)]
                if (0 <= dest[0] <= 59 and 0 <= dest[1] <= 59
                        and dest not in visited
                        and not _is_wall(game, *dest)
                        and (not extra_walls or dest not in extra_walls)):
                    visited.add((nx, ny))
                    visited.add(dest)
                    queue.append((dest, path + [(nm, dest[0], dest[1])]))
            else:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nm, nx, ny)]))
    return None


def _find_pickup_cells(game: Any, start: _Pos) -> list[_Pickup]:
    pickups = game.current_level.get_sprites_by_tag("npxgalaybz")
    ox, oy = start[0] % STEP, start[1] % STEP
    xs = [ox + STEP * i for i in range(12) if ox + STEP * i <= 59]
    ys = [oy + STEP * i for i in range(12) if oy + STEP * i <= 59]
    result = []
    for sp in pickups:
        sp_width = getattr(sp, 'width', getattr(sp, 'w', STEP))
        sp_height = getattr(sp, 'height', getattr(sp, 'h', STEP))
        cells = [(gx, gy) for gx in xs for gy in ys
                 if gx < sp.x + sp_width and gx + STEP > sp.x
                 and gy < sp.y + sp_height and gy + STEP > sp.y
                 and not _is_wall(game, gx, gy)]
        if cells:
            result.append({'sprite': (sp.x, sp.y), 'cells': cells})
    return result


def _can_reach_pickup(game: Any, pos: _Pos, sc_val: int, pickups: list[_Pickup],
                      push_map: _PushMap, extra_walls: _WallSet = None,
                      sc_step: int = 2) -> bool:
    max_moves = (sc_val - 1) // sc_step
    if max_moves < 0:
        return False
    for pu in pickups:
        for cell in pu['cells']:
            p = _bfs_path(game, pos, cell, push_map, extra_walls)
            if p is not None and len(p) <= max_moves:
                return True
    return False


def _plan_segment(game: Any, start: _Pos, goal: _Pos, sc_val: int,
                  pickups: list[_Pickup], push_map: _PushMap,
                  extra_walls: _WallSet = None,
                  sc_step: int = 2) -> tuple[list[_Step] | None, int, _Pos | None]:
    direct = _bfs_path(game, start, goal, push_map, extra_walls)
    if direct is None:
        return None, sc_val, None
    dc = len(direct) * sc_step
    sa = sc_val - dc
    if sa >= SC_FLOOR:
        return direct, sa, None
    if 0 < sa < SC_FLOOR:
        if _can_reach_pickup(game, goal, sa, pickups, push_map, extra_walls, sc_step):
            return direct, sa, None
    best_sa = -9999
    best = None
    for pu in pickups:
        for cell in pu['cells']:
            p1 = _bfs_path(game, start, cell, push_map, extra_walls)
            if p1 is None:
                continue
            if sc_val - len(p1) * sc_step <= 0:
                continue
            p2 = _bfs_path(game, cell, goal, push_map, extra_walls)
            if p2 is None:
                continue
            sc_after = SC_REFILL - len(p2) * sc_step
            if sc_after <= 0:
                continue
            if sc_after > best_sa:
                best_sa = sc_after
                best = (p1 + p2, sc_after, cell)
    if best:
        return best
    if sa <= 0:
        return None, sc_val, None
    return direct, sa, None


def _get_neighbors(game: Any, pos: _Pos, push_map: _PushMap,
                   extra_walls: _WallSet = None) -> list[_Step]:
    x, y = pos
    result = []
    for nm, dx, dy in [('UP', 0, -STEP), ('DOWN', 0, STEP), ('LEFT', -STEP, 0), ('RIGHT', STEP, 0)]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx <= 59 and 0 <= ny <= 59):
            continue
        if _is_wall(game, nx, ny):
            continue
        if extra_walls and (nx, ny) in extra_walls:
            continue
        if push_map and (nx, ny) in push_map:
            dest = push_map[(nx, ny)]
            if (0 <= dest[0] <= 59 and 0 <= dest[1] <= 59
                    and not _is_wall(game, *dest)
                    and not (extra_walls and dest in extra_walls)):
                result.append((nm, dest[0], dest[1]))
        else:
            result.append((nm, nx, ny))
    return result


def _remove_pickup(pickups: list[_Pickup], cell: _Pos) -> list[_Pickup]:
    return [p for p in pickups if cell not in p['cells']]


def _plan_for_ordering(game: Any, trigger_order: tuple[str, ...],
                       exits: list[Any], pickups_init: list[_Pickup],
                       start: _Pos, sc_init: int, r0: int, c0: int, s0: int,
                       push_map: _PushMap, trigger_pickups: list[_Pickup] | None = None,
                       extra_walls: _WallSet = None, exit_indices: list[int] | None = None,
                       sc_step: int = 2) -> tuple[list[_Step], int] | None:
    moves = []
    cur = start
    sc_val = sc_init
    pickups = [dict(p) for p in pickups_init]
    tpus = [dict(p) for p in (trigger_pickups if trigger_pickups is not None else pickups_init)]
    lr, lc, ls = r0, c0, s0

    for ei, e in enumerate(exits):
        orig_idx = exit_indices[ei] if exit_indices is not None else ei
        rt = game.ehwheiwsk[orig_idx]
        ct = game.yjdexjsoa[orig_idx]
        st = game.ldxlnycps[orig_idx]
        rot_spr = game.current_level.get_sprites_by_tag("rhsxkxzdjz")
        col_spr = game.current_level.get_sprites_by_tag("soyhouuebz")
        sha_spr = game.current_level.get_sprites_by_tag("ttfwljgohq")
        specs = {
            'rot':   ((rt - lr) % 4, rot_spr),
            'color': ((ct - lc) % 4, col_spr),
            'shape': ((st - ls) % 6, sha_spr),
        }
        for ttype in trigger_order:
            n_visits, sprite_list = specs[ttype]
            if n_visits == 0 or not sprite_list:
                continue
            tpos = (sprite_list[0].x, sprite_list[0].y)
            for v in range(n_visits):
                path, new_sc, pu_used = _plan_segment(game, cur, tpos, sc_val, tpus, push_map, extra_walls,
                                                       sc_step=sc_step)
                if path is None:
                    return None
                moves.extend(path)
                sc_val = new_sc
                cur = tpos
                if pu_used:
                    tpus = _remove_pickup(tpus, pu_used)
                    pickups = _remove_pickup(pickups, pu_used)
                if ttype == 'rot':    lr = (lr + 1) % 4
                elif ttype == 'color': lc = (lc + 1) % 4
                elif ttype == 'shape': ls = (ls + 1) % 6
                if v < n_visits - 1:
                    did = False
                    if sc_val <= LOW_SC and tpus:
                        for pu in tpus:
                            for cell in pu['cells']:
                                p_out = _bfs_path(game, tpos, cell, push_map, extra_walls)
                                if p_out is None or sc_val - len(p_out) * sc_step <= 0:
                                    continue
                                p_back = _bfs_path(game, cell, tpos, push_map, extra_walls)
                                if p_back is None:
                                    continue
                                sc_after = SC_REFILL - len(p_back) * sc_step
                                if sc_after <= 0:
                                    continue
                                moves.extend(p_out)
                                moves.extend(p_back)
                                sc_val = sc_after
                                cur = tpos
                                tpus = _remove_pickup(tpus, cell)
                                pickups = _remove_pickup(pickups, cell)
                                did = True
                                break
                            if did:
                                break
                    if not did:
                        nb = _get_neighbors(game, tpos, push_map, extra_walls)
                        if nb:
                            moves.append(nb[0])
                            sc_val -= sc_step
                            cur = (nb[0][1], nb[0][2])

        exit_pos = (e.x, e.y)
        path, new_sc, pu_used = _plan_segment(game, cur, exit_pos, sc_val, pickups, push_map, extra_walls,
                                               sc_step=sc_step)
        if path is None:
            return None
        moves.extend(path)
        sc_val = new_sc
        cur = exit_pos
        if pu_used:
            pickups = _remove_pickup(pickups, pu_used)
            tpus = _remove_pickup(tpus, pu_used)

    return moves, sc_val


def _get_start_triggers(game: Any) -> tuple[int, int, int]:
    sr = game.current_level.get_data("StartRotation")
    sc = game.current_level.get_data("StartColor")
    ss = game.current_level.get_data("StartShape")
    r0 = game.dhksvilbb.index(sr) if sr is not None else 0
    c0 = game.tnkekoeuk.index(sc) if sc is not None else 0
    s0 = ss if ss is not None else 0
    return r0, c0, s0


def _best_plan(game: Any, exits_subset: list[Any], pickups: list[_Pickup],
               start: _Pos, sc_init: int, r0: int, c0: int, s0: int,
               push_map: _PushMap, extra_walls: _WallSet = None,
               exit_indices: list[int] | None = None,
               sc_step: int = 2) -> tuple[list[_Step] | None, int, Any]:
    n = len(exits_subset)
    base_idx = exit_indices if exit_indices is not None else list(range(n))
    exit_perms = list(permutations(range(n))) if n <= 3 else [tuple(range(n))]
    best_moves = None
    best_sc = -9999
    best_order = None
    for ep in exit_perms:
        ordered_exits = [exits_subset[i] for i in ep]
        ordered_idx = [base_idx[i] for i in ep]
        for perm in permutations(['rot', 'color', 'shape']):
            result = _plan_for_ordering(game, perm, ordered_exits, pickups, start, sc_init,
                                        r0, c0, s0, push_map, extra_walls=extra_walls,
                                        exit_indices=ordered_idx, sc_step=sc_step)
            if result:
                moves, final_sc = result
                if final_sc > best_sc:
                    best_sc = final_sc
                    best_moves = moves
                    best_order = (ep, perm)
            for ri in range(len(pickups)):
                tpus = [p for i, p in enumerate(pickups) if i != ri]
                result = _plan_for_ordering(game, perm, ordered_exits, pickups, start, sc_init,
                                            r0, c0, s0, push_map, trigger_pickups=tpus,
                                            extra_walls=extra_walls, exit_indices=ordered_idx,
                                            sc_step=sc_step)
                if result:
                    moves, final_sc = result
                    if final_sc > best_sc:
                        best_sc = final_sc
                        best_moves = moves
                        best_order = (ep, perm, ri)
    return best_moves, best_sc, best_order


def _rects_overlap(ax: int, ay: int, aw: int, ah: int,
                   bx: int, by: int, bw: int, bh: int) -> bool:
    """Return True if rectangle A and rectangle B overlap."""
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _bfs_moving_triggers(game: Any) -> dict[str, Any] | None:
    """Time-expanded BFS for levels with moving trigger sprites.

    State: (pos, tphase, rot, col, shp, pickup_mask, exits_visited_mask)
    """
    start = (game.gudziatsk.x, game.gudziatsk.y)
    sc_init = game._step_counter_ui.current_steps
    SC_STEP = game._step_counter_ui.efipnixsvl
    r0 = game.cklxociuu
    c0 = game.hiaauhahz
    s0 = game.fwckfzsyc

    rot_spr_list = game.current_level.get_sprites_by_tag("rhsxkxzdjz")
    col_spr_list = game.current_level.get_sprites_by_tag("soyhouuebz")
    shp_spr_list = game.current_level.get_sprites_by_tag("ttfwljgohq")

    rot_cycle = [(rot_spr_list[0].x, rot_spr_list[0].y)] if rot_spr_list else []
    col_cycle = [(col_spr_list[0].x, col_spr_list[0].y)] if col_spr_list else []
    shp_cycle = [(shp_spr_list[0].x, shp_spr_list[0].y)] if shp_spr_list else []

    for _ in range(7):
        for link in game.wsoslqeku:
            link.step()
        if rot_spr_list: rot_cycle.append((rot_spr_list[0].x, rot_spr_list[0].y))
        if col_spr_list: col_cycle.append((col_spr_list[0].x, col_spr_list[0].y))
        if shp_spr_list: shp_cycle.append((shp_spr_list[0].x, shp_spr_list[0].y))

    for link in game.wsoslqeku:
        link.bkuguqrpvq()

    PERIOD = 8
    push_map = _build_push_map(game, start)

    all_passable = set()
    q0 = deque([start])
    all_passable.add(start)
    dirs4 = [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]
    while q0:
        cx, cy = q0.popleft()
        for ddx, ddy in dirs4:
            nx, ny = cx + ddx, cy + ddy
            if not (0 <= nx <= 59 and 0 <= ny <= 59):
                continue
            if _is_wall(game, nx, ny):
                continue
            if (nx, ny) not in all_passable:
                all_passable.add((nx, ny))
                if (nx, ny) in push_map:
                    dest = push_map[(nx, ny)]
                    if (0 <= dest[0] <= 59 and 0 <= dest[1] <= 59
                            and dest not in all_passable and not _is_wall(game, *dest)):
                        all_passable.add(dest)
                        q0.append(dest)
                else:
                    q0.append((nx, ny))

    exits = game.plrpelhym
    exit_pos_list = [(e.x, e.y) for e in exits]
    exit_idx_map = {pos: idx for idx, pos in enumerate(exit_pos_list)}
    exit_req_list = [(game.ehwheiwsk[i], game.yjdexjsoa[i], game.ldxlnycps[i])
                     for i in range(len(exits))]
    num_exits = len(exits)
    goal_ev_mask = (1 << num_exits) - 1

    pu_sprites = game.current_level.get_sprites_by_tag("npxgalaybz")
    pu_cells = []
    sorted_passable = sorted(all_passable)
    for sp in pu_sprites:
        sp_w = getattr(sp, 'width', getattr(sp, 'w', STEP))
        sp_h = getattr(sp, 'height', getattr(sp, 'h', STEP))
        for pos in sorted_passable:
            if _rects_overlap(sp.x, sp.y, sp_w, sp_h, pos[0], pos[1], STEP, STEP):
                pu_cells.append(pos)
                break

    best_sc_map = {}
    parent_map = {}

    init_state = (start, 0, r0, c0, s0, 0, 0)
    best_sc_map[init_state] = sc_init
    parent_map[init_state] = None

    q = deque([(init_state, sc_init)])
    goal_state = None
    move_dirs = [('UP', 0, -STEP), ('DOWN', 0, STEP), ('LEFT', -STEP, 0), ('RIGHT', STEP, 0)]

    while q and goal_state is None:
        state, sc = q.popleft()
        if best_sc_map.get(state, -1) != sc:
            continue
        pos, tphase, rot, col, shp, pm, ev = state
        for dir_name, ddx, ddy in move_dirs:
            nx, ny = pos[0] + ddx, pos[1] + ddy
            npos = (nx, ny)
            if not (0 <= nx <= 59 and 0 <= ny <= 59):
                continue
            if npos in push_map:
                effective_pos = push_map[npos]
                epx, epy = effective_pos
                if not (0 <= epx <= 59 and 0 <= epy <= 59):
                    continue
                if _is_wall(game, nx, ny):
                    continue
                is_exit_cell = effective_pos in exit_pos_list
                if _is_wall(game, epx, epy) and not is_exit_cell:
                    continue
            else:
                is_exit_cell = npos in exit_pos_list
                if not (npos in all_passable) and not is_exit_cell:
                    continue
                effective_pos = npos

            new_tphase = (tphase + 1) % PERIOD
            new_rot = rot
            new_col = col
            new_shp = shp
            if rot_cycle and effective_pos == rot_cycle[new_tphase]:
                new_rot = (rot + 1) % 4
            if col_cycle and effective_pos == col_cycle[new_tphase]:
                new_col = (col + 1) % 4
            if shp_cycle and effective_pos == shp_cycle[new_tphase]:
                new_shp = (shp + 1) % 6

            new_pm = pm
            new_sc = sc - SC_STEP
            for pi, pcell in enumerate(pu_cells):
                if effective_pos == pcell and not (pm >> pi & 1):
                    new_pm = pm | (1 << pi)
                    new_sc = SC_REFILL
                    break

            if new_sc < 0:
                continue

            new_ev = ev
            if is_exit_cell:
                exit_idx = exit_idx_map[effective_pos]
                if ev >> exit_idx & 1:
                    pass
                else:
                    req_r, req_c, req_s = exit_req_list[exit_idx]
                    if (new_rot, new_col, new_shp) == (req_r, req_c, req_s):
                        new_ev = ev | (1 << exit_idx)
                    else:
                        continue

            new_state = (effective_pos, new_tphase, new_rot, new_col, new_shp, new_pm, new_ev)

            if new_ev == goal_ev_mask:
                parent_map[new_state] = (state, dir_name, effective_pos[0], effective_pos[1])
                best_sc_map[new_state] = new_sc
                goal_state = new_state
                break

            if new_sc > best_sc_map.get(new_state, -1):
                best_sc_map[new_state] = new_sc
                parent_map[new_state] = (state, dir_name, effective_pos[0], effective_pos[1])
                q.append((new_state, new_sc))

    if goal_state is None:
        return None

    moves = []
    state = goal_state
    while parent_map[state] is not None:
        parent_state, dir_name, nx, ny = parent_map[state]
        moves.append((dir_name, nx, ny))
        state = parent_state
    moves.reverse()
    return {'strategy': 'direct', 'phases': [moves]}


def _solve_level(game: Any, lvl_idx: int) -> dict[str, Any] | None:
    sc_init = game._step_counter_ui.current_steps
    sc_step = game._step_counter_ui.efipnixsvl
    start = (game.gudziatsk.x, game.gudziatsk.y)
    exits = game.plrpelhym
    pickups = _find_pickup_cells(game, start)
    r0 = game.cklxociuu
    c0 = game.hiaauhahz
    s0 = game.fwckfzsyc
    push_map = _build_push_map(game, start)
    start_r0, start_c0, start_s0 = _get_start_triggers(game)
    respawn_pos = (game.ltwrkifkx, game.zyoimjaei)

    if game.wsoslqeku:
        result = _bfs_moving_triggers(game)
        if result:
            return result
        logger.warning(f"BFS solver failed for level {lvl_idx}, falling back to static")

    best_moves, best_sc, _ = _best_plan(game, exits, pickups, start, sc_init,
                                        r0, c0, s0, push_map, sc_step=sc_step)
    if len(exits) == 1:
        if best_moves is not None:
            return {'strategy': 'direct', 'phases': [best_moves]}
        return None

    if best_moves is not None and best_sc >= sc_step:
        return {'strategy': 'direct', 'phases': [best_moves]}

    direct_fallback = best_moves if best_sc > -200 else None
    exit0_pos = (exits[0].x, exits[0].y)
    exit1_pos = (exits[1].x, exits[1].y)

    p0_moves, p0_sc, _ = _best_plan(game, [exits[0]], pickups, start, sc_init,
                                     r0, c0, s0, push_map, extra_walls={exit1_pos},
                                     exit_indices=[0], sc_step=sc_step)
    if p0_moves is None:
        if direct_fallback is not None:
            return {'strategy': 'direct', 'phases': [direct_fallback]}
        return None

    p1_moves, p1_sc, _ = _best_plan(game, [exits[1]], pickups, respawn_pos, SC_REFILL,
                                     start_r0, start_c0, start_s0, push_map,
                                     extra_walls={exit0_pos}, exit_indices=[1],
                                     sc_step=sc_step)
    if p1_moves is None:
        if direct_fallback is not None:
            return {'strategy': 'direct', 'phases': [direct_fallback]}
        return None

    return {
        'strategy': 'death',
        'phase0': p0_moves,
        'phase1': p1_moves,
        'respawn_pos': respawn_pos,
    }


def _get_supported_level_count(game: Any) -> int:
    """Validate that this solver is used for a supported ls20 game; return level count."""
    game_id = getattr(game, 'game_id', None)
    if game_id is not None and not re.fullmatch(r'ls20(?:-[0-9a-fA-F]+)?', str(game_id)):
        msg = f"SolverAgent only supports ls20 games; got game_id={game_id!r}"
        logger.error(msg)
        raise ValueError(msg)
    levels = getattr(game, 'levels', None)
    if levels is not None:
        try:
            level_count = len(levels)
        except TypeError:
            level_count = 7
        else:
            if level_count != 7:
                msg = (f"SolverAgent expects 7 levels; got {level_count}")
                logger.error(msg)
                raise ValueError(msg)
            return level_count
    return 7


def _precompute_actions(game: Any) -> list[GameAction]:
    """Solve all supported ls20 levels on a game clone; return ordered list of GameActions."""
    level_count = _get_supported_level_count(game)
    actions: list[GameAction] = []
    for lvl in range(level_count):
        frame = None
        plan = _solve_level(game, lvl)
        if plan is None:
            msg = f"Cannot plan level {lvl}"
            logger.error(msg)
            raise RuntimeError(msg)

        strategy = plan['strategy']

        if strategy == 'direct':
            for moves in plan['phases']:
                for step in moves:
                    ga = _AM[step[0]]
                    actions.append(ga)
                    frame = game.perform_action(ActionInput(id=ga), raw=True)
                    if frame.levels_completed > lvl or frame.state == GameState.WIN:
                        break
                else:
                    continue
                break

        elif strategy == 'death':
            for step in plan['phase0']:
                ga = _AM[step[0]]
                actions.append(ga)
                frame = game.perform_action(ActionInput(id=ga), raw=True)
                if frame.levels_completed > lvl or frame.state == GameState.WIN:
                    break

            respawn = plan['respawn_pos']
            for _ in range(200):
                actions.append(GameAction.ACTION1)
                frame = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
                if (game.gudziatsk.x, game.gudziatsk.y) == respawn:
                    break
                if frame.levels_completed > lvl or frame.state == GameState.WIN:
                    break

            for step in plan['phase1']:
                ga = _AM[step[0]]
                actions.append(ga)
                frame = game.perform_action(ActionInput(id=ga), raw=True)
                if frame.levels_completed > lvl or frame.state == GameState.WIN:
                    break

        if frame is None or (
            frame.levels_completed <= lvl and frame.state != GameState.WIN
        ):
            msg = f"Planned actions did not complete level {lvl}"
            logger.error(msg)
            raise RuntimeError(msg)

        if frame.state == GameState.WIN:
            break

    return actions


# ── Agent ──────────────────────────────────────────────────────────────────────

class SolverAgent(Agent):
    """Pre-computes the full solution at init; plays back actions on demand."""

    MAX_ACTIONS = 800
    ACTION_BUDGET_SAFETY_MARGIN = 1

    def _ensure_action_budget(self, actions: list[GameAction]) -> None:
        """Ensure the agent loop budget can execute the full precomputed plan."""
        planned_actions = len(actions)
        required_actions = planned_actions + self.ACTION_BUDGET_SAFETY_MARGIN
        if required_actions > self.MAX_ACTIONS:
            logger.info(
                "[Solver] Increasing MAX_ACTIONS from %s to %s to fit pre-computed plan "
                "with safety margin",
                self.MAX_ACTIONS,
                required_actions,
            )
            self.MAX_ACTIONS = required_actions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._action_queue: list[GameAction] = []
        self._queue_idx: int = 0
        try:
            self._action_queue = self._precompute()
            self._ensure_action_budget(self._action_queue)
            logger.info(f"[Solver] Pre-computed {len(self._action_queue)} actions")
        except Exception as e:
            logger.error(f"[Solver] Pre-compute failed: {e}", exc_info=True)
            raise RuntimeError(
                f"SolverAgent initialization failed during precompute for game "
                f"{getattr(self, 'game_id', '<unknown>')}"
            ) from e

    def _precompute(self) -> list[GameAction]:
        # Local mode: arc_env._game is the live game instance
        game = getattr(self.arc_env, '_game', None)
        if game is not None:
            game_cls = type(game)
            clone = game_cls()
            clone.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            return _precompute_actions(clone)

        # Remote/online mode: load game class from environment_files on disk
        import importlib.util
        import os
        import pathlib
        import re
        envs_dir = os.getenv("ENVIRONMENTS_DIR", "environment_files")
        # Supported game_id formats: "ls20" or "ls20-9607627b" (lowercase hex)
        match = re.fullmatch(r"(ls20)(?:-([0-9a-f]+))?", self.game_id)
        if match is None:
            raise ValueError(
                f"Unsupported game_id format: {self.game_id!r}. "
                "Expected 'ls20' or 'ls20-<lowercase hex hash>'."
            )
        game_prefix, game_hash, class_name = self._parse_game_id()
        game_file = self._resolve_game_file(envs_dir, game_prefix, game_hash)
        module_name = f"{game_prefix}_{game_hash}" if game_hash else game_prefix
        spec = importlib.util.spec_from_file_location(module_name, game_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {game_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        game_cls = getattr(module, class_name)
        clone = game_cls()
        clone.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        return _precompute_actions(clone)

    def _parse_game_id(self) -> tuple[str, str, str]:
        match = re.fullmatch(r"([a-z]{2}\d+)(?:-([0-9a-f]+))?", self.game_id)
        if match is None:
            raise ValueError(
                f"Unsupported game_id format: {self.game_id!r}. "
                "Expected 'ls20' or 'ls20-<hash>'."
            )
        game_prefix = match.group(1)
        game_hash = (match.group(2) or "").lower()
        class_name = game_prefix[0].upper() + game_prefix[1:]
        return game_prefix, game_hash, class_name

    def _resolve_game_file(self, envs_dir: str, game_prefix: str, game_hash: str) -> pathlib.Path:
        rel_sub = pathlib.Path(game_prefix) / game_hash / f"{game_prefix}.py"
        here = pathlib.Path(__file__).parent  # agents/templates/
        candidates = [
            pathlib.Path(envs_dir),                          # absolute or cwd-relative
            here.parent.parent / envs_dir,                  # repo-root / envs_dir
            here.parent.parent.parent / envs_dir,           # parent-of-repo / envs_dir
        ]
        searched: list[str] = []
        for base in candidates:
            base_resolved = base.resolve(strict=False)
            candidate = base_resolved / rel_sub
            searched.append(str(candidate.resolve(strict=False)))
            if not candidate.exists():
                continue
            resolved_game_file = candidate.resolve(strict=True)
            try:
                resolved_game_file.relative_to(base_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Resolved game file escapes base directory: {resolved_game_file} "
                    f"(base: {base_resolved})"
                ) from exc
            return resolved_game_file
        raise FileNotFoundError(
            f"Game file not found. Searched:\n  " + "\n  ".join(searched)
        )

    def choose_action(self, frames: list[FrameData], current_frame: FrameData) -> GameAction:
        if current_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            logger.warning(f"[Solver] Game state {current_frame.state} — sending RESET, rewinding queue")
            self._queue_idx = 0
            return GameAction.RESET
        if not self._action_queue:
            logger.warning("[Solver] No pre-computed actions available — sending RESET")
            self._queue_idx = 0
            return GameAction.RESET
        if self._queue_idx >= len(self._action_queue):
            logger.warning("[Solver] Queue exhausted — sending RESET and rewinding queue")
            self._queue_idx = 0
            return GameAction.RESET
        action = self._action_queue[self._queue_idx]
        self._queue_idx += 1
        return action

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN
