
/**
 * Chapter 5 — Landsat 8 fire-severity indices
 *


// 1. Image acquisition dates
var prefire_start = '2019-05-18';
var prefire_end = '2019-05-24';

var postfire_start = '2020-06-05';
var postfire_end = '2020-06-09';


// 2. Landsat 8 image composites
function composite(start, end, cloudLimit) {
  return ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    .filterDate(start, end)
    .filterBounds(roi)
    .filter(ee.Filter.lt('CLOUD_COVER', cloudLimit))
    .select(['B5', 'B7'])
    .median();
}

var pre = composite(prefire_start, prefire_end, 20);
var post = composite(postfire_start, postfire_end, 30);


// 3. Normalised Burn Ratio (NBR)
var preNBR = pre
  .normalizedDifference(['B5', 'B7'])
  .rename('preNBR');

var postNBR = post
  .normalizedDifference(['B5', 'B7'])
  .rename('postNBR');


// 4. Differenced Normalised Burn Ratio (dNBR)
var dNBR = preNBR
  .subtract(postNBR)
  .rename('dNBR');


// 5. Background dNBR offset
// Mean dNBR from the external unburned reference area.
var offset = ee.Number(
  dNBR.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: fire.geometry(),
    scale: 30,
    maxPixels: 1e9
  }).get('dNBR')
);

var dNBR_offset = dNBR
  .subtract(offset)
  .rename('dNBR_offset');


// 6. Relative differenced Normalised Burn Ratio (RdNBR)
var RdNBR = dNBR_offset
  .divide(
    preNBR.abs().sqrt().max(0.000001)
  )
  .rename('RdNBR');


// 7. Map visualisation
Map.centerObject(roi, 10);

Map.addLayer(
  dNBR_offset.multiply(1000),
  {min: -1000, max: 2000},
  'dNBR offset x1000'
);

Map.addLayer(
  RdNBR,
  {min: -1, max: 1},
  'RdNBR'
);


// 8. Export fire-severity indices
[
  ['preNBR', preNBR],
  ['postNBR', postNBR],
  ['dNBR_unscaled', dNBR],
  ['dNBR_offset', dNBR_offset],
  ['RdNBR', RdNBR]
].forEach(function(item) {
  Export.image.toDrive({
    image: item[1],
    description: item[0],
    scale: 30,
    region: roi.geometry(),
    maxPixels: 1e13
  });
});
