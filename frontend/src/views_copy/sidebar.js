import React from 'react'


import './sidebar.css'

const Sidebar = (props) => {
  return (
    <div className="sidebar-container">
        <title>exported project</title>
      <div className="sidebar-sidebar">
        <div className="sidebar-task-option">
          <span className="sidebar-text-main">
            <span>Main</span>
          </span>
          <div className="sidebar-option-main">
            <div className="sidebar-frame-task-option">
              <span className="sidebar-text-task-option">Task Option</span>
              <div className="sidebar-mode-card">
                <div className="sidebar-frame-opencoding">
                  <div className="sidebar-checkbox">
                    <input
                      type="radio"
                      name="radio"
                      className="sidebar-radiobutton"
                    />
                  </div>
                  <div className="sidebar-choice-opencoding">
                    <span className="sidebar-text01">Open Coding</span>
                    <span className="sidebar-text02">
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
                <div className="sidebar-frame-sumsakey">
                  <div className="sidebar-checkbox1">
                    <input
                      type="radio"
                      name="radio"
                      className="sidebar-radiobutton1"
                    />
                  </div>
                  <div className="sidebar-choice-sumsakey">
                    <span className="sidebar-text06">SUM / S.A / K.E</span>
                    <span className="sidebar-text07">
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
            <div className="sidebar-frame-model-option">
              <select className="sidebar-select">
                <option value="trinity-1.2B">trinity-1.2B</option>
                <option value="ko-gpt2">ko-gpt2</option>
                <option value="polyglot-ko-1.3b">polyglot-ko-1.3b</option>
              </select>
            </div>
            <div className="sidebar-option-bar-graph">
              <div className="sidebar-checkbox2">
                <input type="checkbox" checked="true" />
              </div>
              <span className="sidebar-text24">
                Show Bar Graph (opencoding-only)
              </span>
            </div>
            <div className="sidebar-option-pie-chart">
              <div className="sidebar-checkbox4">
                <input type="checkbox" checked="true" />
              </div>
              <span className="sidebar-text25">
                Show Pie Chart (sentiment analysis-only)
              </span>
            </div>
          </div>
        </div>
        <div className="sidebar-history-main">
          <div className="sidebar-frame117">
            <div className="sidebar-segmentcell">
              <img
                alt="collectionI117"
                src="/external/collectioni117-kzbr.svg"
                className="sidebar-collection"
              />
              <span className="sidebar-text26">
                <span>History</span>
              </span>
            </div>
          </div>
          <div className="sidebar-container1">
            <div className="sidebar-segment">
              <div className="sidebar-frame118">
                <div className="sidebar-segmentcell1">
                  <div className="sidebar-frame119">
                    <img
                      alt="chatalt2I117"
                      src="/external/chatalt2i117-fyzc.svg"
                      className="sidebar-chatalt2"
                    />
                    <span className="sidebar-text28">
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
