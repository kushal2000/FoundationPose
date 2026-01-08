import trimesh
import math


mallet_dimensions = {
    "HAMMER_HANDLE_LENGTH": 0.24,
    "HAMMER_HANDLE_WIDTH": 0.03,
    "HAMMER_HANDLE_THICKNESS": 0.02,
    "HAMMER_HEAD_LENGTH": 0.08,
    "HAMMER_HEAD_WIDTH": 0.05,
    "HAMMER_HEAD_THICKNESS": 0.045
}


hammer_dimensions = {
    "HAMMER_HANDLE_LENGTH": 0.25,
    "HAMMER_HANDLE_WIDTH": 0.03,
    "HAMMER_HANDLE_THICKNESS": 0.02,
    "HAMMER_HEAD_LENGTH": 0.11,
    "HAMMER_HEAD_WIDTH": 0.02,
    "HAMMER_HEAD_THICKNESS": 0.02
}

def create_obj_from_dimensions_dict(dimensions_dict):
    handle = trimesh.primitives.Box(
        extents=[dimensions_dict["HAMMER_HANDLE_LENGTH"], dimensions_dict["HAMMER_HANDLE_WIDTH"], dimensions_dict["HAMMER_HANDLE_THICKNESS"]]
    )
    head = trimesh.primitives.Box(
        extents=[dimensions_dict["HAMMER_HEAD_WIDTH"], dimensions_dict["HAMMER_HEAD_LENGTH"], dimensions_dict["HAMMER_HEAD_THICKNESS"]]
    )
    head.apply_translation(
        [dimensions_dict["HAMMER_HANDLE_LENGTH"] / 2.0 + dimensions_dict["HAMMER_HEAD_WIDTH"] / 2.0, 0.0, 0.0]
    )
    return trimesh.util.concatenate([handle, head])

mallet = create_obj_from_dimensions_dict(mallet_dimensions)
mallet.export('new_objects/cuboidal_mallet.obj')
mallet.export('new_objects/cuboidal_mallet.stl')
hammer = create_obj_from_dimensions_dict(hammer_dimensions)
hammer.export('new_objects/cuboidal_hammer.obj')
hammer.export('new_objects/cuboidal_hammer.stl')