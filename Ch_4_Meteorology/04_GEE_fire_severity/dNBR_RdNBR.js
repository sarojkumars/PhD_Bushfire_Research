// Requires imported Earth Engine assets StdArea and fire (offset buffer).
// Historical Landsat Collection 1 script; see README before running.


// Study area - 2009 black saturday fire
// import study area shapefile (in zip format) and use it as fire extent as a table  (import as a cloud assets from asset tab)
// Name the variable to StdArea 

// visualise Study Area
Map.centerObject(StdArea, 11);
Map.addLayer(StdArea, {color: 'grey'}, 'Study area');

// Landsat 5 TM images for pre-fire and post-fire dates
// Images were manually checked for cloud cover

var prefire_start = '2008-02-27';   
var prefire_end = '2008-03-05';

var postfire_start = '2009-04-19';
var postfire_end = '2009-04-25';

//prefire image collection 
var PreFire = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    .filterDate(prefire_start, prefire_end)
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
    .select(['B1','B2','B3','B4','B5','B7'])
    .filterBounds(StdArea);
    
print(PreFire)    

//postfire image collection 
var PostFire = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
    .filterDate(postfire_start, postfire_end)
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
    .select(['B1','B2','B3','B4','B5','B7'])
    .filterBounds(StdArea);
    
print(PostFire) 

//Compute median of imagery for each band
var premedian = PreFire.median();
var postmedian = PostFire.median();

// Calculate NBR for pre- and post-fire images 
// NBR = (NIR-SWIR) / (NIR+SWIR)

var preNBR = premedian.normalizedDifference(['B4', 'B7']);
var postNBR = postmedian.normalizedDifference(['B4', 'B7']);

var indices = preNBR.addBands(postNBR)

print("indices", indices);
print("PreNBR : ", preNBR);
print("PostNBR: ", postNBR);

//differenced NBR preFire and postFire
var dNBR_unscaled = preNBR.subtract(postNBR);

print("dNBR_unscaled: ", dNBR_unscaled);

// Scaling (dnbr * 1000) - for calculation simplicity as suggested by Key and Benson 2006
var dNBR = dNBR_unscaled.multiply(1000).rename('dnbr2').toFloat().addBands(indices);

// Add image
print("dNBR_scaled: ", dNBR.select('dnbr2'));

// calcuate dNBR offset
// calculate dNBR offset using parks et al. i.e., by using ring buffer of 180m outside fire
// the buffer layer is added as an asset

var dNBR2 = ee.Image.constant(ee.Number(dNBR.select('dnbr2').reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: fire.geometry(),
    scale: 30,
    maxPixels: 1e9
  }).get('dnbr2'))).rename('offset').toFloat().addBands(dNBR);

var dNBR3 = dNBR2.expression(
          "b('dnbr2') - b('offset')").
          rename('dNBR_offset').toFloat().addBands(dNBR2);
          

var dNBR_off = dNBR3.select('dNBR_offset')
print("dNBR_offset: ", dNBR3.select('dNBR_offset'))

// for rdnbr

var sqrt_abs_nbr = preNBR.abs().sqrt();
var RdNBR = dNBR3.expression(
  "(b('dNBR_offset')/1000)").divide(sqrt_abs_nbr);

print("RdNBR: ", RdNBR);

// Define visualization parameters
var visParams = {min: -1, max: 1, palette: ['brown', 'white', 'green']};

// Define visualization parameters
var visParams2 = {min: -1000, max: 2000, palette: ['brown', 'white', 'green']};

// Add dNBR layer to the map
Map.addLayer(dNBR_off, visParams2, 'dNBR_off');

// Add RdNBR layer to the map
Map.addLayer(RdNBR, visParams, 'RdNBR');

// Add dNBR_unscaled layer to the map
Map.addLayer(dNBR_unscaled, visParams, 'dNBR_unscaled');


// Add preNBR layer to the map
Map.addLayer(preNBR, visParams, 'preNBR');

// Add postNBR layer to the map
Map.addLayer(postNBR, visParams, 'postNBR');

// Export images. We can also define crs and projection below. Otherwise default projectin - wgs84.
Export.image.toDrive({
  image: preNBR,
  description: 'preNBR',
  //crs: projection.crs,
  //crsTransform: projection.transform,
  scale:30,
  region: StdArea
});

Export.image.toDrive({
  image: postNBR,
  description: 'postNBR',
  //crs: projection.crs,
  //crsTransform: projection.transform,
  scale:30,
  region: StdArea
});

Export.image.toDrive({
  image: dNBR_unscaled,
  description: 'dNBR_unscaled',
  //crs: projection.crs,
  //crsTransform: projection.transform,
  scale:30,
  region: StdArea
});

Export.image.toDrive({
  image: dNBR_off,
  description: 'dNBR_offset',
  //crs: projection.crs,
  //crsTransform: projection.transform,
  scale:30,
  region: StdArea
});

Export.image.toDrive({
  image: RdNBR,
  description: 'RdNBR',
  //crs: projection.crs,
  //crsTransform: projection.transform,
  scale:30,
  region: StdArea
});

// the indices are stored in google drive.. 












