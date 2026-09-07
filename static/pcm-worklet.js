class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.port.onmessage = (event) => {
      if (event.data === "stop") this.active = false;
    };
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (this.active && channel?.length) {
      const copy = new Float32Array(channel);
      this.port.postMessage(copy, [copy.buffer]);
    }
    return this.active;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
