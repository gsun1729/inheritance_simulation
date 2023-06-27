from configs.structural_config import (MOTHER_RADIUS,
                                        MOTHER_SURFACE_AREA,
                                        MOTHER_VOLUME,
                                        PARTICLE_RADIUS,
                                        PARTICLE_VOLUME,
                                        N_PARTICLES,
                                        MOTHER_BOX_SIDELENGTH,
                                        PARTICLE_CROSSAREA,
                                        MOTHER_MITO_VOL_PERCENT,
                                        MOTHER_MITO_TOTAL_VOL,
                                        AGGREGATE_PERCENT)


if __name__ == "__main__":
    print(f"Mother Radius :\t{MOTHER_RADIUS}")
    print(f"Mother Sphere SA :\t{MOTHER_SURFACE_AREA}")
    print(f"Mother Sphere Vol :\t{MOTHER_VOLUME}")
    print(f"Particle radius :\t{PARTICLE_RADIUS}")
    print(f"Particle sphere volume :\t{PARTICLE_VOLUME}")
    print(f"N_particles :\t{N_PARTICLES}")
    print(f"Mother sidelength :\t{MOTHER_BOX_SIDELENGTH}")
    print(
        f"Percentage of 2D box occupied by particle cross sectional area :\t{(N_PARTICLES*PARTICLE_CROSSAREA)/(MOTHER_BOX_SIDELENGTH**2)}")
    print(
        f"Percentage of 3D mom sphere occupied by particle 3D vol :\t{(N_PARTICLES*PARTICLE_VOLUME)/(MOTHER_VOLUME)}")

    print(f"Percent mito by volume :\t{MOTHER_MITO_VOL_PERCENT}")
    print(f"Mother mito total vol :\t{MOTHER_MITO_TOTAL_VOL}")
    print(f"Aggregate Percentage :\t{AGGREGATE_PERCENT}")
    print(f"nAggregates :\t{int(AGGREGATE_PERCENT * N_PARTICLES)}")
