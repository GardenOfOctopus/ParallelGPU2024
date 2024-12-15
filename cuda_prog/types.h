
struct Bl_coords{
        int x; int y; int z;
};

struct Bl_size{
        int x; int y; int z;
};

struct Bl_iterations{
        int i_begin; int i_end; 
        int j_begin; int j_end; 
        int k_begin; int k_end;
};

enum BoundaryType{
        NORTH,
        EAST, 
        SOUTH, 
	    WEST,
        TOP,
        BOTTOM, 
};
