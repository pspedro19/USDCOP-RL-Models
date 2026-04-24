import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setConcurrency(null);
Config.setCodec("h264");
Config.setCrf(17);
Config.setPixelFormat("yuv420p");
Config.setAudioCodec("aac");
Config.setAudioBitrate("320k");
Config.setChromiumOpenGlRenderer("angle");
Config.setOverwriteOutput(true);
Config.setPublicDir("public");
Config.setEntryPoint("./src/index.ts");
