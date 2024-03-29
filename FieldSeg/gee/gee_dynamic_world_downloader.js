var research_area = ee.Geometry.Polygon([
           [116.33571944444445, 35.71913333333333],
           [116.71186666666667, 35.65492777777778],
           [116.62324444444444, 35.34498611111111],
           [116.24454166666666, 35.421419444444446]]);
research_area = table
var color_palette = ["FFFFFF","CE7E45","DF923D","F1B555","FCD163","99B718","74A901","66A000","529400","3E8601","207401","056201","004C00","023B01","012E01","011D01","011301"]

/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */

// function maskS2clouds(image) {
//   var qa = image.select('QA60');
//   // Bits 10 and 11 are clouds and cirrus, respectively.
//   var cloudBitMask = 1 << 10;
//   var cirrusBitMask = 1 << 11;
//   // Both flags should be set to zero, indicating clear conditions.
//   var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
//       .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
//   return image.updateMask(mask).divide(10000);
// }

// function applyVegetationIndex(image) {
//   var redBands = image.select('B4');
//   var nirBands = image.select('B8');
//   var ndvi = nirBands.subtract(redBands).divide(nirBands.add(redBands)).rename("NDVI");
//   var nocloud = image.select('MSK_CLDPRB').multiply(-1).rename("Clear")
//   return image.addBands(ndvi, null, true).addBands(nocloud, null, true);
// }

// var dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') // 
//                   .filterBounds(research_area)
//                   .filter(ee.Filter.calendarRange(5, 9, "month"))
//                   .filterDate('2020-01-01', '2021-01-01')
//                   .sort('CLOUDY_PIXEL_PERCENTAGE', false)
//                   .map(function(image){
//                     return image.addBands(image.metadata('system:time_start'));
//                   })
//                   .map(maskS2clouds)
//                   .mosaic();

var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
var dwCol = dwCol
              .filterBounds(research_area)
              .filter(ee.Filter.calendarRange(5, 9, "month"))
              .filterDate('2019-01-01', '2022-01-01')
              .select('crops')
              .mean()

dwCol = dwCol.clip(research_area)
var mask = dwCol.lte(0.99);
dwCol = dwCol.updateMask(mask).unmask(0.99)
mask = dwCol.gte(0.01);
dwCol = dwCol.updateMask(mask).unmask(0.01)
dwCol = dwCol.multiply(255).toInt8()
Map.addLayer(dwCol, {"bands":["crops"], "min":1, "max":255, "palette":color_palette}, 'crop prob');
// Map.addLayer(mask, {"min":0, "max":1}, 'mask');

Export.image.toDrive({
          image: dwCol,
          region: research_area,
          scale: 10, 
          description: "DynamicWorld_ChinaMainLand",
          maxPixels:1e13
        });

// var visualization = {
//   min: 0.0,
//   max: 0.3,
//   bands: ['B4', 'B3', 'B2'],
// };
// Map.setCenter(116, 35, 15);
// Map.addLayer(dataset, visualization, 'rgb');


// Export.image.toDrive({
//           image: dataset.select(['B8', 'B4', 'B3', 'B2']),
//           region: research_area,
//           scale: 10, 
//           description: "Sentinel_TaiAn",
//           maxPixels:1e13
//         });


