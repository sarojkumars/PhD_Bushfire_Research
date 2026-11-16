// =====================================================================
// 1. SETTINGS
// =====================================================================


var fireArea = studyArea.geometry();

var analysisScale = 100;
var exportCRS = 'EPSG:3577';
var minimumObservations = 2;
var recoveryThreshold = 0.80;

// Apply background/phenological NBR offset.
var applyNBROffset = true;

Map.centerObject(fireArea, 7);


// =====================================================================
// 2. TEMPORAL SETTINGS
// =====================================================================

// Common seasonal window: 1 February to 30 April.
//
// Five pre-fire seasonal composites:
// 2015, 2016, 2017, 2018, 2019
//
// PRE baseline = pixel-wise median of these five annual composites.
// PRE2019 is retained separately for sensitivity analysis.
//
// POST = 2020
// T1   = 2021
// T2   = 2022
// T3   = 2023
// T4   = 2024
// T5   = 2025

var prefireYears = [
  2015,
  2016,
  2017,
  2018,
  2019
];


var recoveryWindows = [
  {
    state: 'POST',
    yearIndex: 0,
    start: '2020-02-01',
    end:   '2020-05-01'
  },
  {
    state: 'T1',
    yearIndex: 1,
    start: '2021-02-01',
    end:   '2021-05-01'
  },
  {
    state: 'T2',
    yearIndex: 2,
    start: '2022-02-01',
    end:   '2022-05-01'
  },
  {
    state: 'T3',
    yearIndex: 3,
    start: '2023-02-01',
    end:   '2023-05-01'
  },
  {
    state: 'T4',
    yearIndex: 4,
    start: '2024-02-01',
    end:   '2024-05-01'
  },
  {
    state: 'T5',
    yearIndex: 5,
    start: '2025-02-01',
    end:   '2025-05-01'
  }
];


// =====================================================================
// 3. LANDSAT COLLECTIONS
// =====================================================================

var landsat8 = ee.ImageCollection(
  'LANDSAT/LC08/C02/T1_L2'
);

var landsat9 = ee.ImageCollection(
  'LANDSAT/LC09/C02/T1_L2'
);


// =====================================================================
// 4. LANDSAT PREPROCESSING
// =====================================================================

function prepareLandsat(image) {

  var qa = image.select('QA_PIXEL');

  // Remove:
  // fill, dilated cloud, cirrus, cloud, cloud shadow, snow, water.
  var clearMask = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 1).eq(0))
    .and(qa.bitwiseAnd(1 << 2).eq(0))
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0))
    .and(qa.bitwiseAnd(1 << 7).eq(0));

  var saturationMask = image
    .select('QA_RADSAT')
    .eq(0);

  // Landsat Collection 2 Level-2 scale/offset.
  var optical = image
    .select(
      [
        'SR_B2',
        'SR_B4',
        'SR_B5',
        'SR_B6',
        'SR_B7'
      ],
      [
        'blue',
        'red',
        'nir',
        'swir1',
        'swir2'
      ]
    )
    .multiply(0.0000275)
    .add(-0.2)
    .toFloat();


  var nbr = optical
    .normalizedDifference([
      'nir',
      'swir2'
    ])
    .rename('NBR')
    .toFloat();


  var ndvi = optical
    .normalizedDifference([
      'nir',
      'red'
    ])
    .rename('NDVI')
    .toFloat();


  var evi = optical.expression(
    '2.5 * ((NIR - RED) / ' +
    '(NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
    {
      NIR: optical.select('nir'),
      RED: optical.select('red'),
      BLUE: optical.select('blue')
    }
  )
    .rename('EVI')
    .toFloat();


  return ee.Image.cat([
      nbr,
      ndvi,
      evi
    ])
    .updateMask(clearMask)
    .updateMask(saturationMask)
    .toFloat()
    .copyProperties(
      image,
      ['system:time_start']
    );
}


// =====================================================================
// 5. MERGED LANDSAT 8 + LANDSAT 9 COLLECTION
// =====================================================================

var landsat = landsat8
  .merge(landsat9)
  .filterBounds(fireArea)
  .map(prepareLandsat);


// =====================================================================
// 6. SEASONAL COMPOSITE FUNCTION
// =====================================================================

function seasonalComposite(
  startDate,
  endDate,
  stateName,
  yearIndex
) {

  var collection = landsat
    .filterDate(startDate, endDate);

  var validCount = collection
    .select('NBR')
    .count()
    .rename('valid_count')
    .toFloat();

  var composite = collection
    .median()
    .select([
      'NBR',
      'NDVI',
      'EVI'
    ])
    .toFloat()
    .updateMask(
      validCount.gte(minimumObservations)
    )
    .addBands(validCount)
    .toFloat();

  return composite.set({
    state: stateName,
    year_index: yearIndex,
    start_date: startDate,
    end_date: endDate,
    image_count: collection.size(),
    'system:time_start':
      ee.Date(startDate).millis()
  });
}


// =====================================================================
// 7. FIVE-YEAR PRE-FIRE SEASONAL BASELINE
// =====================================================================

var prefireAnnualImages =
  prefireYears.map(function(year) {

    var startDate =
      ee.Date.fromYMD(year, 2, 1);

    var endDate =
      ee.Date.fromYMD(year, 5, 1);

    return seasonalComposite(
      startDate,
      endDate,
      'PRE_' + year,
      -1
    ).set(
      'baseline_year',
      year
    );
  });


var prefireAnnualCollection =
  ee.ImageCollection.fromImages(
    prefireAnnualImages
  );


print(
  'Pre-fire annual seasonal composites',
  prefireAnnualCollection
);


print(
  'Pre-fire image counts',
  prefireAnnualCollection
    .aggregate_array('image_count')
);


// Each year contributes one seasonal composite;
// the five annual composites are then combined using the median.

var preBaseline = prefireAnnualCollection
  .select([
    'NBR',
    'NDVI',
    'EVI'
  ])
  .median()
  .rename([
    'NBR',
    'NDVI',
    'EVI'
  ])
  .toFloat()
  .set({
    state: 'PRE_BASELINE',
    year_index: -1
  });


// Retain immediate pre-fire 2019 seasonal composite
// for sensitivity analysis.

var pre2019 = ee.Image(
  prefireAnnualCollection
    .filter(
      ee.Filter.eq(
        'baseline_year',
        2019
      )
    )
    .first()
).toFloat();


// =====================================================================
// 8. POST-FIRE AND RECOVERY COMPOSITES
// =====================================================================

var postfireImages =
  recoveryWindows.map(function(window) {

    return seasonalComposite(
      window.start,
      window.end,
      window.state,
      window.yearIndex
    );
  });


var postfireCollection =
  ee.ImageCollection.fromImages(
    postfireImages
  );


print(
  'Post-fire annual composites',
  postfireCollection
);


print(
  'Post-fire image counts',
  postfireCollection
    .aggregate_array('image_count')
);


// =====================================================================
// 9. EXTERIOR REFERENCE RINGS
// =====================================================================

// IMPORTANT:
// fireArea should represent the actual fire/burn perimeter.
//
// Primary phenological/background correction:
// 0–180 m outside fire perimeter.

var exterior180 =
  fireArea.buffer(180);

var offsetRing = exterior180
  .difference(
    fireArea,
    30
  );


// Backup ring:
//
// If the 180 m ring has no valid observations for a composite,
// a broader 0–1000 m exterior ring is used.

var exterior1000 =
  fireArea.buffer(1000);

var backupOffsetRing =
  exterior1000
    .difference(
      fireArea,
      30
    );


Map.addLayer(
  offsetRing,
  {color: '00FFFF'},
  '180 m offset ring',
  false
);


// =====================================================================
// 10. NBR BACKGROUND / PHENOLOGICAL OFFSET
// =====================================================================

// For each annual image:
//
// background dNBR = PRE baseline NBR - annual NBR
//
// Offset = mean background dNBR in nearby unburned reference area.
//
// Corrected NBR = annual NBR + offset.
//
// This offset correction is applied to NBR only.
//
// NDVI and EVI remain seasonally matched supporting indicators.


function getOffsetValue(
  dNBRImage,
  geometry
) {

  var result =
    dNBRImage.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geometry,
      scale: 90,
      bestEffort: true,
      maxPixels: 1e9,
      tileScale: 8
    });

  return result.get(
    'background_dNBR'
  );
}


function correctNBR(image) {

  image = ee.Image(image);

  var backgroundDNBR =
    preBaseline
      .select('NBR')
      .subtract(
        image.select('NBR')
      )
      .rename(
        'background_dNBR'
      )
      .toFloat();


  var offset180 =
    getOffsetValue(
      backgroundDNBR,
      offsetRing
    );


  var offsetBackup =
    getOffsetValue(
      backgroundDNBR,
      backupOffsetRing
    );


  var offset = ee.Number(
    ee.Algorithms.If(
      ee.Algorithms.IsEqual(
        offset180,
        null
      ),
      ee.Algorithms.If(
        ee.Algorithms.IsEqual(
          offsetBackup,
          null
        ),
        0,
        offsetBackup
      ),
      offset180
    )
  );


  var corrected = ee.Image(
    ee.Algorithms.If(
      applyNBROffset,
      image
        .select('NBR')
        .add(offset),
      image
        .select('NBR')
    )
  )
    .rename('NBR_corrected')
    .toFloat();


  return image
    .addBands(corrected)
    .toFloat()
    .set({
      NBR_offset: offset
    });
}


var correctedPostfireCollection =
  postfireCollection.map(
    correctNBR
  );


print(
  'Annual NBR background offsets',
  correctedPostfireCollection
    .aggregate_array(
      'NBR_offset'
    )
);


// =====================================================================
// 11. GET TEMPORAL STATES
// =====================================================================

function getImage(stateName) {

  return ee.Image(
    correctedPostfireCollection
      .filter(
        ee.Filter.eq(
          'state',
          stateName
        )
      )
      .first()
  ).toFloat();
}


var post = getImage('POST');


// =====================================================================
// 12. NORMALISED RECOVERY
// =====================================================================

// R(t) =
// [Index(t) - Index(POST)]
// ------------------------
// [Index(PRE) - Index(POST)]
//
// Interpretation:
//
// 0   = immediate post-fire condition
// 1   = return to multi-year pre-fire baseline
// >1  = exceeds pre-fire baseline
// <0  = further departure from baseline


var minimumNBRDenominator = 0.05;
var minimumVIDenominator = 0.05;


var nbrDenominator =
  preBaseline
    .select('NBR')
    .subtract(
      post.select('NBR_corrected')
    )
    .toFloat();


var ndviDenominator =
  preBaseline
    .select('NDVI')
    .subtract(
      post.select('NDVI')
    )
    .toFloat();


var eviDenominator =
  preBaseline
    .select('EVI')
    .subtract(
      post.select('EVI')
    )
    .toFloat();


function addRecovery(image) {

  image = ee.Image(image);


  var nbrRecovery = image
    .select('NBR_corrected')
    .subtract(
      post.select('NBR_corrected')
    )
    .divide(nbrDenominator)
    .updateMask(
      nbrDenominator
        .abs()
        .gte(
          minimumNBRDenominator
        )
    )
    .rename(
      'NBR_recovery'
    )
    .toFloat();


  var ndviRecovery = image
    .select('NDVI')
    .subtract(
      post.select('NDVI')
    )
    .divide(ndviDenominator)
    .updateMask(
      ndviDenominator
        .abs()
        .gte(
          minimumVIDenominator
        )
    )
    .rename(
      'NDVI_recovery'
    )
    .toFloat();


  var eviRecovery = image
    .select('EVI')
    .subtract(
      post.select('EVI')
    )
    .divide(eviDenominator)
    .updateMask(
      eviDenominator
        .abs()
        .gte(
          minimumVIDenominator
        )
    )
    .rename(
      'EVI_recovery'
    )
    .toFloat();


  var yearBand =
    ee.Image.constant(
      ee.Number(
        image.get(
          'year_index'
        )
      )
    )
    .rename(
      'year_index'
    )
    .toFloat();


  return image
    .addBands([
      nbrRecovery,
      ndviRecovery,
      eviRecovery,
      yearBand
    ])
    .toFloat()
    .copyProperties(
      image,
      image.propertyNames()
    );
}


var recoveryCollection =
  correctedPostfireCollection
    .map(addRecovery);


var recoveryYears =
  recoveryCollection
    .filter(
      ee.Filter.gt(
        'year_index',
        0
      )
    );


function getRecoveryState(
  stateName
) {

  return ee.Image(
    recoveryCollection
      .filter(
        ee.Filter.eq(
          'state',
          stateName
        )
      )
      .first()
  ).toFloat();
}


var rT1 =
  getRecoveryState('T1');

var rT2 =
  getRecoveryState('T2');

var rT3 =
  getRecoveryState('T3');

var rT4 =
  getRecoveryState('T4');

var rT5 =
  getRecoveryState('T5');


// =====================================================================
// 13. RECOVERY MAGNITUDE
// =====================================================================

// Recovery magnitude here represents the observed recovery state at T5.

var recoveryMagnitude =
  rT5
    .select([
      'NBR_recovery',
      'NDVI_recovery',
      'EVI_recovery'
    ])
    .rename([
      'NBR_magnitude_T5',
      'NDVI_magnitude_T5',
      'EVI_magnitude_T5'
    ])
    .toFloat();


// =====================================================================
// 14. RECOVERY RATE
// =====================================================================

// Pixel-wise linear recovery rate over T1-T5.
//
// This provides a summary trajectory metric.
// The complete annual trajectory should still be retained
// for subsequent statistical analysis.


function calculateRate(
  responseBand,
  outputBand
) {

  var regressionCollection =
    recoveryYears.map(
      function(image) {

        return ee.Image(image)
          .select([
            'year_index',
            responseBand
          ])
          .toFloat();
      }
    );


  return regressionCollection
    .reduce(
      ee.Reducer.linearFit()
    )
    .select('scale')
    .rename(outputBand)
    .toFloat();
}


var nbrRecoveryRate =
  calculateRate(
    'NBR_recovery',
    'NBR_recovery_rate'
  );


var ndviRecoveryRate =
  calculateRate(
    'NDVI_recovery',
    'NDVI_recovery_rate'
  );


var eviRecoveryRate =
  calculateRate(
    'EVI_recovery',
    'EVI_recovery_rate'
  );


// =====================================================================
// 15. NBR RECOVERY PERSISTENCE
// =====================================================================

// Persistence = proportion of T1-T5 observations
// with NBR recovery >= 80%.

var nbrRecoveryStack =
  ee.Image.cat([

    rT1.select(
      'NBR_recovery'
    ).rename('T1'),

    rT2.select(
      'NBR_recovery'
    ).rename('T2'),

    rT3.select(
      'NBR_recovery'
    ).rename('T3'),

    rT4.select(
      'NBR_recovery'
    ).rename('T4'),

    rT5.select(
      'NBR_recovery'
    ).rename('T5')

  ])
  .toFloat();


var yearsRecovered =
  nbrRecoveryStack
    .gte(
      recoveryThreshold
    )
    .reduce(
      ee.Reducer.sum()
    )
    .toFloat();


var recoveryPersistence80 =
  yearsRecovered
    .divide(5.0)
    .rename(
      'NBR_persistence_080'
    )
    .toFloat();


// Complementary measure:
// fraction of annual stages below 80%.

var persistenceBelow80 =
  ee.Image.constant(1.0)
    .toFloat()
    .subtract(
      recoveryPersistence80
    )
    .rename(
      'NBR_fraction_below_080'
    )
    .toFloat();


// =====================================================================
// 16. SUSTAINED TIME TO 80% RECOVERY
// =====================================================================

// First annual stage at which recovery reaches >= 80%
// AND remains >= 80% in all following annual observations.
//
// Codes:
//
// 0 = sustained recovery not reached by T5
// 1 = sustained from T1
// 2 = sustained from T2
// 3 = sustained from T3
// 4 = sustained from T4
// 5 = threshold reached at T5


var above1 =
  rT1.select(
    'NBR_recovery'
  ).gte(
    recoveryThreshold
  );


var above2 =
  rT2.select(
    'NBR_recovery'
  ).gte(
    recoveryThreshold
  );


var above3 =
  rT3.select(
    'NBR_recovery'
  ).gte(
    recoveryThreshold
  );


var above4 =
  rT4.select(
    'NBR_recovery'
  ).gte(
    recoveryThreshold
  );


var above5 =
  rT5.select(
    'NBR_recovery'
  ).gte(
    recoveryThreshold
  );


var sustainedFromT1 =
  above1
    .and(above2)
    .and(above3)
    .and(above4)
    .and(above5);


var sustainedFromT2 =
  above2
    .and(above3)
    .and(above4)
    .and(above5);


var sustainedFromT3 =
  above3
    .and(above4)
    .and(above5);


var sustainedFromT4 =
  above4
    .and(above5);


var sustainedFromT5 =
  above5;


var timeToRecovery80 =
  ee.Image.constant(0)
    .toFloat()

    .where(
      sustainedFromT5,
      5
    )

    .where(
      sustainedFromT4,
      4
    )

    .where(
      sustainedFromT3,
      3
    )

    .where(
      sustainedFromT2,
      2
    )

    .where(
      sustainedFromT1,
      1
    )

    .rename(
      'NBR_sustained_TTR_080'
    )

    .updateMask(
      rT5
        .select(
          'NBR_recovery'
        )
        .mask()
    )

    .toFloat();


// =====================================================================
// 17. PRE-FIRE BASELINE SENSITIVITY
// =====================================================================

// Compare the immediate 2019 pre-fire seasonal condition
// against the 2015-2019 multi-year seasonal baseline.

var pre2019Sensitivity =
  ee.Image.cat([

    pre2019
      .select('NBR')
      .subtract(
        preBaseline.select(
          'NBR'
        )
      )
      .rename(
        'PRE2019_minus_baseline_NBR'
      ),

    pre2019
      .select('NDVI')
      .subtract(
        preBaseline.select(
          'NDVI'
        )
      )
      .rename(
        'PRE2019_minus_baseline_NDVI'
      ),

    pre2019
      .select('EVI')
      .subtract(
        preBaseline.select(
          'EVI'
        )
      )
      .rename(
        'PRE2019_minus_baseline_EVI'
      )

  ])
  .toFloat();


// =====================================================================
// 18. RAW SPECTRAL INDEX STACK
// =====================================================================

var rawIndexStack =
  ee.Image.cat([

    // Multi-year baseline
    preBaseline
      .select('NBR')
      .rename(
        'PREbaseline_NBR'
      ),

    preBaseline
      .select('NDVI')
      .rename(
        'PREbaseline_NDVI'
      ),

    preBaseline
      .select('EVI')
      .rename(
        'PREbaseline_EVI'
      ),


    // Immediate 2019 pre-fire condition
    pre2019
      .select('NBR')
      .rename(
        'PRE2019_NBR'
      ),

    pre2019
      .select('NDVI')
      .rename(
        'PRE2019_NDVI'
      ),

    pre2019
      .select('EVI')
      .rename(
        'PRE2019_EVI'
      ),


    // Immediate post-fire
    post
      .select(
        'NBR_corrected'
      )
      .rename(
        'POST_NBR'
      ),

    post
      .select('NDVI')
      .rename(
        'POST_NDVI'
      ),

    post
      .select('EVI')
      .rename(
        'POST_EVI'
      ),


    // T1
    rT1
      .select(
        'NBR_corrected'
      )
      .rename(
        'T1_NBR'
      ),

    rT1
      .select('NDVI')
      .rename(
        'T1_NDVI'
      ),

    rT1
      .select('EVI')
      .rename(
        'T1_EVI'
      ),


    // T2
    rT2
      .select(
        'NBR_corrected'
      )
      .rename(
        'T2_NBR'
      ),

    rT2
      .select('NDVI')
      .rename(
        'T2_NDVI'
      ),

    rT2
      .select('EVI')
      .rename(
        'T2_EVI'
      ),


    // T3
    rT3
      .select(
        'NBR_corrected'
      )
      .rename(
        'T3_NBR'
      ),

    rT3
      .select('NDVI')
      .rename(
        'T3_NDVI'
      ),

    rT3
      .select('EVI')
      .rename(
        'T3_EVI'
      ),


    // T4
    rT4
      .select(
        'NBR_corrected'
      )
      .rename(
        'T4_NBR'
      ),

    rT4
      .select('NDVI')
      .rename(
        'T4_NDVI'
      ),

    rT4
      .select('EVI')
      .rename(
        'T4_EVI'
      ),


    // T5
    rT5
      .select(
        'NBR_corrected'
      )
      .rename(
        'T5_NBR'
      ),

    rT5
      .select('NDVI')
      .rename(
        'T5_NDVI'
      ),

    rT5
      .select('EVI')
      .rename(
        'T5_EVI'
      )

  ])
  .clip(fireArea)
  .toFloat();


// =====================================================================
// 19. NORMALISED RECOVERY STACK
// =====================================================================

var recoveryStack =
  ee.Image.cat([

    rT1
      .select([
        'NBR_recovery',
        'NDVI_recovery',
        'EVI_recovery'
      ])
      .rename([
        'T1_NBR_recovery',
        'T1_NDVI_recovery',
        'T1_EVI_recovery'
      ]),

    rT2
      .select([
        'NBR_recovery',
        'NDVI_recovery',
        'EVI_recovery'
      ])
      .rename([
        'T2_NBR_recovery',
        'T2_NDVI_recovery',
        'T2_EVI_recovery'
      ]),

    rT3
      .select([
        'NBR_recovery',
        'NDVI_recovery',
        'EVI_recovery'
      ])
      .rename([
        'T3_NBR_recovery',
        'T3_NDVI_recovery',
        'T3_EVI_recovery'
      ]),

    rT4
      .select([
        'NBR_recovery',
        'NDVI_recovery',
        'EVI_recovery'
      ])
      .rename([
        'T4_NBR_recovery',
        'T4_NDVI_recovery',
        'T4_EVI_recovery'
      ]),

    rT5
      .select([
        'NBR_recovery',
        'NDVI_recovery',
        'EVI_recovery'
      ])
      .rename([
        'T5_NBR_recovery',
        'T5_NDVI_recovery',
        'T5_EVI_recovery'
      ])

  ])
  .clip(fireArea)
  .toFloat();


// =====================================================================
// 20. TRAJECTORY METRIC STACK
// =====================================================================

var trajectoryMetrics =
  ee.Image.cat([

    recoveryMagnitude,

    nbrRecoveryRate,
    ndviRecoveryRate,
    eviRecoveryRate,

    recoveryPersistence80,
    persistenceBelow80,

    timeToRecovery80,

    pre2019Sensitivity

  ])
  .clip(fireArea)
  .toFloat();


// =====================================================================
// 21. MAP VISUALISATION
// =====================================================================

var recoveryVis = {
  min: 0,
  max: 1,
  palette: [
    '7F0000',
    'D73027',
    'FC8D59',
    'FEE08B',
    'D9EF8B',
    '91CF60',
    '1A9850',
    '006837'
  ]
};


Map.addLayer(
  preBaseline.select('NBR'),
  {
    min: -0.3,
    max: 0.8,
    palette: [
      '7F0000',
      'FEE08B',
      '006837'
    ]
  },
  'PRE baseline NBR',
  false
);


Map.addLayer(
  pre2019.select('NBR'),
  {
    min: -0.3,
    max: 0.8,
    palette: [
      '7F0000',
      'FEE08B',
      '006837'
    ]
  },
  'PRE 2019 NBR',
  false
);


Map.addLayer(
  rT1.select(
    'NBR_recovery'
  ),
  recoveryVis,
  'NBR recovery T1',
  false
);


Map.addLayer(
  rT2.select(
    'NBR_recovery'
  ),
  recoveryVis,
  'NBR recovery T2',
  false
);


Map.addLayer(
  rT3.select(
    'NBR_recovery'
  ),
  recoveryVis,
  'NBR recovery T3',
  false
);


Map.addLayer(
  rT4.select(
    'NBR_recovery'
  ),
  recoveryVis,
  'NBR recovery T4',
  false
);


Map.addLayer(
  rT5.select(
    'NBR_recovery'
  ),
  recoveryVis,
  'NBR recovery T5',
  true
);


Map.addLayer(
  nbrRecoveryRate,
  {
    min: -0.1,
    max: 0.3,
    palette: [
      'B2182B',
      'F7F7F7',
      '2166AC'
    ]
  },
  'NBR recovery rate',
  false
);


Map.addLayer(
  recoveryPersistence80,
  {
    min: 0,
    max: 1,
    palette: [
      '7F0000',
      'FEE08B',
      '006837'
    ]
  },
  'NBR recovery persistence >=80%',
  false
);


Map.addLayer(
  persistenceBelow80,
  {
    min: 0,
    max: 1,
    palette: [
      '006837',
      'FEE08B',
      'B2182B'
    ]
  },
  'Fraction below 80% recovery',
  false
);


Map.addLayer(
  timeToRecovery80,
  {
    min: 0,
    max: 5,
    palette: [
      '808080',
      '006837',
      '91CF60',
      'FEE08B',
      'FC8D59',
      'D73027'
    ]
  },
  'Sustained time to 80% recovery',
  false
);


// =====================================================================
// 22. QUALITY CONTROL
// =====================================================================

print(
  'PRE baseline',
  preBaseline
);

print(
  'PRE 2019',
  pre2019
);

print(
  'POST',
  post
);

print(
  'NBR denominator',
  nbrDenominator
);

print(
  'Recovery states',
  recoveryCollection
    .aggregate_array(
      'state'
    )
);

print(
  'NBR offsets',
  correctedPostfireCollection
    .aggregate_array(
      'NBR_offset'
    )
);

print(
  'Raw stack band types',
  rawIndexStack.bandTypes()
);

print(
  'Recovery stack band types',
  recoveryStack.bandTypes()
);

print(
  'Trajectory metric band types',
  trajectoryMetrics.bandTypes()
);


// =====================================================================
// 23. EXPORT RAW SPECTRAL INDEX STACK
// =====================================================================

Export.image.toDrive({
  image: rawIndexStack.toFloat(),
  description:
    'BlackSummer_Annual_Spectral_Indices',
  folder:
    'BlackSummer_Recovery',
  fileNamePrefix:
    'black_summer_annual_spectral_indices',
  region: fireArea,
  scale: analysisScale,
  crs: exportCRS,
  maxPixels: 1e13,
  fileDimensions: 4096,
  skipEmptyTiles: true,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});


// =====================================================================
// 24. EXPORT NORMALISED RECOVERY STACK
// =====================================================================

Export.image.toDrive({
  image: recoveryStack.toFloat(),
  description:
    'BlackSummer_Normalised_Recovery',
  folder:
    'BlackSummer_Recovery',
  fileNamePrefix:
    'black_summer_normalised_recovery',
  region: fireArea,
  scale: analysisScale,
  crs: exportCRS,
  maxPixels: 1e13,
  fileDimensions: 4096,
  skipEmptyTiles: true,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});


// =====================================================================
// 25. EXPORT TRAJECTORY METRICS
// =====================================================================

Export.image.toDrive({
  image: trajectoryMetrics.toFloat(),
  description:
    'BlackSummer_Recovery_Trajectory_Metrics',
  folder:
    'BlackSummer_Recovery',
  fileNamePrefix:
    'black_summer_recovery_trajectory_metrics',
  region: fireArea,
  scale: analysisScale,
  crs: exportCRS,
  maxPixels: 1e13,
  fileDimensions: 4096,
  skipEmptyTiles: true,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});


print(
  'Processing complete. Review QC values and start export tasks.'
);