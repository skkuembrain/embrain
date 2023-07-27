import React from 'react';

// import { Helmet } from 'react-helmet'

import './sidebar.css';

const Sidebar = (props) => {
  return (
    <div className="sidebar-container">
      {/* <Helmet> */}
        <title>exported project</title>
      {/* </Helmet> */}
      <div className="sidebar-sidebar">
        <div className="sidebar-frame164">
          <span className="sidebar-text">
            <span>Main</span>
          </span>
          <div className="sidebar-frame139">
            <div className="sidebar-frame155">
              <span className="sidebar-text02">Task Option</span>
              <div className="sidebar-mode-card">
                <div className="sidebar-frame136">
                  <div className="sidebar-checkbox">
                    <img
                      alt="image_1"
                      src="/external/checkbox.svg"
                      className="sidebar-image"
                    />
                  </div>
                  <div className="sidebar-frame137">
                    <span className="sidebar-text03">Open Coding</span>
                    <span className="sidebar-text04">
                      <span>
                        간단한 리뷰를 넣으면,
                        <span
                          dangerouslySetInnerHTML={{
                            __html: ' ',
                          }}
                        />
                      </span>
                      <br></br>
                      <span>&lt;속성, 의견&gt;형식으로 추출합니다.</span>
                    </span>
                  </div>
                </div>
              </div>
              <div className="sidebar-mode-card1">
                <div className="sidebar-frame1361">
                  <img
                    alt="CheckboxI117"
                    src="/external/checkboxi117-04xj.svg"
                    className="sidebar-checkbox1"
                  />
                  <div className="sidebar-frame1371">
                    <span className="sidebar-text08">SUM / S.A / K.E</span>
                    <span className="sidebar-text09">
                      <span>Summary</span>
                      <br></br>
                      <span>문장 형태로 간단히 요약해줍니다.</span>
                      <br></br>
                      <br></br>
                      <span>Sentiment Analysis</span>
                      <br></br>
                      <span>
                        감성 분석을 해줍니다.
                        <span
                          dangerouslySetInnerHTML={{
                            __html: ' ',
                          }}
                        />
                      </span>
                      <br></br>
                      <span>긍정과 부정으로 분리해줍니다.</span>
                      <br></br>
                      <br></br>
                      <span>Keywords Extraction</span>
                      <br></br>
                      <span>명사구 형태로 키워드를 뽑아줍니다.</span>
                      <br></br>
                    </span>
                  </div>
                </div>
              </div>
            </div>
            <div className="sidebar-frame149">
              <span className="sidebar-text26">skt/trinity-1.2B</span>
              <img
                alt="chevrondownI117"
                src="/external/chevrondowni117-31vd.svg"
                className="sidebar-chevrondown"
              />
            </div>
            <div className="sidebar-frame144">
              <img
                alt="CheckboxI117"
                src="/external/checkboxi117-017.svg"
                className="sidebar-checkbox2"
              />
              <span className="sidebar-text27">
                Show Bar Graph (opencoding-only)
              </span>
            </div>
            <div className="sidebar-frame156">
              <img
                alt="CheckboxI117"
                src="/external/checkboxi117-1k5b.svg"
                className="sidebar-checkbox3"
              />
              <span className="sidebar-text28">
                Show Pie Chart (sentiment analysis-only)
              </span>
            </div>
            <div className="sidebar-frame146">
              <div className="sidebar-checkbox4">
                <img
                  alt="image_2"
                  src="/external/checkbox.svg"
                  className="sidebar-image1"
                />
              </div>
              <span className="sidebar-text29">
                <span>Dark mode</span>
              </span>
            </div>
          </div>
        </div>
        <div className="sidebar-container1">
          <div className="sidebar-frame117">
            <div className="sidebar-segmentcell">
              <img
                alt="collectionI117"
                src="/external/collectioni117-kzbr.svg"
                className="sidebar-collection"
              />
              <span className="sidebar-text31">
                <span>History</span>
              </span>
            </div>
          </div>
          <div className="sidebar-container2">
            <div className="sidebar-segment">
              <div className="sidebar-frame118">
                <div className="sidebar-segmentcell1">
                  <div className="sidebar-frame119">
                    <img
                      alt="chatalt2I117"
                      src="/external/chatalt2i117-fyzc.svg"
                      className="sidebar-chatalt2"
                    />
                    <span className="sidebar-text33">
                      <span>Main</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sidebar
