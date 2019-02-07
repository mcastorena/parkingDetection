from parkingDetector import parkingDetector

vid = "lotVid.avi"
xml = "carclassifier.xml"
yml = "parking2.yml"
json = "parkingData.json"

myDetector = parkingDetector(vid, xml, yml, json)
myDetector.run()