
Dataset link - https://asdc.larc.nasa.gov/data/ACCLIP/Model_WB57/Data_1/

Downloading code - 
```bash
URL=<enter the top level URL here> 
TOKEN=<paste your token here>
wget --header "Authorization: Bearer $TOKEN" --recursive --no-parent --reject "index.html*" --execute robots=off $URL
```

### Main CSV labels
Time_Start, T_GEOS, U_GEOS, V_GEOS, H_GEOS, POTT_GEOS, EPV_GEOS, QV_GEOS, RH_GEOS, O3_GEOS, CO_GEOS, SO2_GEOS, SO4_GEOS, BC_GEOS, OC_GEOS, CONBGL_GEOS, CONBAS_GEOS, CONBEU_GEOS, CONBNA_GEOS, COBBGL_GEOS, COBBAE_GEOS, COBBNA_GEOS, DU_GEOS, TROPPB_GEOS, SLP_GEOS, PS_GEOS.

### Labels meaning
The terms listed are likely data variables related to meteorological and atmospheric measurements, commonly used in scientific research and environmental monitoring. Here's a brief description of each:

1. **Time_Start**: The time at which the measurement was taken, usually in seconds since a reference time (e.g., 00:00:00 UTC on the day of takeoff).

2. **T_GEOS**: Temperature (GEOS t) – Air temperature measured by the GEOS system.

3. **U_GEOS**: U-component of wind (GEOS u) – The eastward component of wind speed measured by GEOS.

4. **V_GEOS**: V-component of wind (GEOS v) – The northward component of wind speed measured by GEOS.

5. **H_GEOS**: Geopotential height (GEOS h) – The height of a pressure level in the atmosphere.

6. **POTT_GEOS**: Potential temperature – Temperature a parcel of air would have if brought to a reference pressure (measured by GEOS).

7. **EPV_GEOS**: Ertel's potential vorticity (GEOS epv) – A measure of the rotation of air parcels, which helps in understanding atmospheric dynamics.

8. **QV_GEOS**: Specific humidity (GEOS qv) – The amount of water vapor in the air per unit mass of air.

9. **RH_GEOS**: Relative humidity (GEOS rh) – The percentage of moisture in the air relative to the maximum amount of moisture the air can hold at that temperature.

10. **O3_GEOS**: Ozone concentration (GEOS o3) – The amount of ozone in the atmosphere.

11. **CO_GEOS**: Carbon monoxide mixing ratio (GEOS co) – The concentration of carbon monoxide in the atmosphere.

12. **SO2_GEOS**: Sulfur dioxide concentration (GEOS so2) – The amount of sulfur dioxide present in the atmosphere.

13. **SO4_GEOS**: Sulfate aerosol concentration (GEOS so4) – The amount of sulfate particles in the air.

14. **BC_GEOS**: Black carbon mass mixing ratio (GEOS bc) – The concentration of black carbon particles in the atmosphere.

15. **OC_GEOS**: Organic carbon mass mixing ratio (GEOS oc) – The amount of organic carbon particles in the air.

16. **CONBGL_GEOS**: CO global non-biomass burning (GEOS conbgl) – Carbon monoxide concentration from global sources excluding biomass burning.

17. **CONBAS_GEOS**: CO Asia non-biomass burning (GEOS conbas) – Carbon monoxide concentration from non-biomass burning sources in Asia.

18. **CONBEU_GEOS**: CO Europe non-biomass burning (GEOS conbeu) – Carbon monoxide concentration from non-biomass burning sources in Europe.

19. **CONBNA_GEOS**: CO North America non-biomass burning (GEOS conbna) – Carbon monoxide concentration from non-biomass burning sources in North America.

20. **COBBGL_GEOS**: CO global biomass burning (GEOS cobbgl) – Carbon monoxide concentration from global biomass burning sources.

21. **COBBAE_GEOS**: CO Asia and Europe biomass burning (GEOS cobbae) – Carbon monoxide concentration from biomass burning in Asia and Europe.

22. **COBBNA_GEOS**: CO North America biomass burning (GEOS cobbna) – Carbon monoxide concentration from biomass burning in North America.

23. **DU_GEOS**: Dust mass mixing ratio (GEOS du) – The concentration of dust particles in the atmosphere.

24. **TROPPB_GEOS**: Tropopause pressure (GEOS troppb) – The atmospheric pressure at the tropopause.

25. **SLP_GEOS**: Mean sea level pressure (GEOS slp) – The atmospheric pressure at sea level.

26. **PS_GEOS**: Surface pressure (GEOS ps) – The atmospheric pressure at the surface level.

These variables are typically used in atmospheric science to study weather patterns, climate change, and air quality.